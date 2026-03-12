"""
Search 에이전트 (Supervisor)

역할:
  1. searchCriteria 기반으로 국내 반도체 스타트업 탐색
  2. 각 기업에 대해 market_eval_agent, tech_summary_agent 작업 요청
  3. 10개 기업의 결과를 모두 수집
  4. 다음 단계 에이전트에 전달 (미구현)
"""
import json
from datetime import datetime, timezone
from typing import TypedDict

from openai import OpenAI

from .market_eval_agent import run_market_eval_agent, MarketEvalState
from .tech_summary_agent import run_tech_summary_agent, TechSummaryState
from .startup_eval_agent import run_startup_eval_agent, StartupEvalState


# ──────────────────────────────────────────
# State 정의
# ──────────────────────────────────────────

class FounderProfile(TypedDict):
    name: str
    role: str
    education: str
    priorCompanies: list[str]
    domainYears: int
    hasExitExperience: bool


class Team(TypedDict):
    founderCount: int
    founderProfiles: list[FounderProfile]


class Funding(TypedDict):
    totalFunding: str       # 예: "50억원", "미공개"
    latestRound: str        # 예: "Seed", "Pre-A", "Series A"
    latestValuation: str    # 예: "200억원", "미공개"
    keyInvestors: list[str]


class Traction(TypedDict):
    revenueYear: list[str]  # 연도별 매출 요약, 예: ["2023: 5억", "2024: 12억"]
    arrGrowthRate: str      # 예: "240%", "미공개"
    keyCustomers: list[str]


class StartupInfo(TypedDict):
    startupId: str
    name: str
    foundedYear: str
    domain: str
    location: str
    stage: str
    team: Team
    funding: Funding
    traction: Traction


class SearchCriteria(TypedDict):
    targetDomain: str       # 예: "AI반도체"
    targetStage: str        # 예: "Seed", "Series A", "" (전체)
    targetRegion: str       # 예: "서울", "판교", "" (전체)
    fetchCount: int         # 고정: 10
    excludeList: list[str]  # 중복 방지용 기업명 또는 ID 목록


class SearchInput(TypedDict):
    searchCriteria: SearchCriteria


class SearchOutput(TypedDict):
    startupList: list[StartupInfo]
    fetchedAt: str      # ISO 8601, 재탐색 시 이전 결과와 비교용
    totalFetched: int   # 실제로 LLM이 가져온 기업 수
    analyses: dict      # 기업명 → {market_eval: MarketEvalOutput, tech_summary: TechSummaryOutput}


class SearchAgentState(TypedDict):
    input: SearchInput
    output: SearchOutput


# ──────────────────────────────────────────
# Supervisor tool 정의
# ──────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_market",
            "description": (
                "market_eval_agent를 호출하여 특정 반도체 스타트업의 시장성을 평가합니다. "
                "marketScore(1~10), scoringReason, sources를 반환합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "startup": {
                        "type": "object",
                        "description": "평가할 스타트업 정보",
                        "properties": {
                            "startupId":     {"type": "string"},
                            "name":          {"type": "string"},
                            "domain":        {"type": "string"},
                            "targetSegment": {"type": "string"},
                            "location":      {"type": "string"},
                            "stage":         {"type": "string"},
                            "funding":       {"type": "object"},
                            "traction":      {"type": "object"},
                        },
                        "required": ["startupId", "name", "domain"],
                    }
                },
                "required": ["startup"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_technology",
            "description": (
                "tech_summary_agent를 호출하여 특정 반도체 스타트업의 기술을 분석합니다. "
                "techScore(1~10), scoringReason, sources를 반환합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "startup": {
                        "type": "object",
                        "description": "기술 분석할 스타트업 정보",
                        "properties": {
                            "startupId": {"type": "string"},
                            "name":      {"type": "string"},
                            "domain":    {"type": "string"},
                        },
                        "required": ["startupId", "name", "domain"],
                    }
                },
                "required": ["startup"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_startup",
            "description": (
                "startup_eval_agent를 호출하여 팀·투자·트랙션을 종합 평가합니다. "
                "finalScore(1~10), scoringReason, sources를 반환합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "startup": {
                        "type": "object",
                        "description": "종합 평가할 스타트업 정보",
                        "properties": {
                            "startupId": {"type": "string"},
                            "name":      {"type": "string"},
                            "team":      {"type": "object"},
                            "funding":   {"type": "object"},
                            "traction":  {"type": "object"},
                        },
                        "required": ["startupId", "name", "team", "funding", "traction"],
                    }
                },
                "required": ["startup"],
            },
        },
    },
]


# ──────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────

def _get_startup_list(criteria: SearchCriteria) -> list[StartupInfo]:
    """searchCriteria 기반으로 국내 반도체 스타트업 목록을 수집합니다."""
    client = OpenAI()

    domain_filter  = criteria.get("targetDomain", "반도체")
    stage_filter   = criteria.get("targetStage", "")
    region_filter  = criteria.get("targetRegion", "")
    fetch_count    = criteria.get("fetchCount", 10)
    exclude_list   = criteria.get("excludeList", [])

    filter_desc = f"- 도메인: {domain_filter}\n"
    if stage_filter:
        filter_desc += f"- 투자 단계: {stage_filter}\n"
    if region_filter:
        filter_desc += f"- 지역: {region_filter}\n"
    if exclude_list:
        filter_desc += f"- 제외 기업: {', '.join(exclude_list)}\n"

    schema_example = json.dumps({
        "startupId": "startup_001",
        "name": "회사명",
        "foundedYear": "2020",
        "domain": "AI반도체",
        "location": "서울",
        "stage": "Series A",
        "team": {
            "founderCount": 3,
            "founderProfiles": [{
                "name": "홍길동",
                "role": "CEO",
                "education": "KAIST 전기전자공학 박사",
                "priorCompanies": ["삼성전자", "퀄컴"],
                "domainYears": 15,
                "hasExitExperience": False
            }]
        },
        "funding": {
            "totalFunding": "50억원",
            "latestRound": "Series A",
            "latestValuation": "300억원",
            "keyInvestors": ["카카오벤처스", "소프트뱅크벤처스"]
        },
        "traction": {
            "revenueYear": ["2023: 5억원", "2024: 12억원"],
            "arrGrowthRate": "140%",
            "keyCustomers": ["현대자동차", "LG전자"]
        }
    }, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 국내 반도체 스타트업 생태계 전문가입니다. "
                    "스타트업의 팀 구성, 투자 현황, 트랙션 정보를 정확히 파악하고 있습니다. "
                    "정보가 불확실한 경우 '미공개' 또는 빈 리스트로 표기하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"다음 조건에 맞는 국내 반도체 스타트업 {fetch_count}개를 탐색해주세요.\n\n"
                    f"## 탐색 조건\n{filter_desc}\n"
                    f"## 응답 형식\n"
                    f"아래 JSON 스키마를 가진 배열로만 응답하세요 (다른 텍스트 없이):\n"
                    f"{schema_example}\n\n"
                    f"위 스키마를 가진 JSON 배열로 {fetch_count}개 기업을 응답하세요."
                ),
            },
        ],
    )

    text = response.choices[0].message.content or ""
    start, end = text.find('['), text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError(f"스타트업 목록 JSON 파싱 실패. 응답: {text[:300]}")

    return json.loads(text[start:end])


def _route_tool(tool_name: str, tool_input: dict) -> str:
    """tool_use 요청을 해당 worker agent로 라우팅합니다."""
    startup = tool_input.get("startup", {})

    if tool_name == "evaluate_market":
        market_state = MarketEvalState(
            input={
                "startupId":     startup["startupId"],
                "domain":        startup["domain"],
                "targetSegment": startup.get("targetSegment", startup["domain"]),
            },
            output={},  # type: ignore[typeddict-item]
        )
        result = run_market_eval_agent(market_state)["output"]

    elif tool_name == "summarize_technology":
        tech_state = TechSummaryState(
            input={
                "startupId": startup["startupId"],
                "name":      startup["name"],
                "domain":    startup["domain"],
            },
            output={},  # type: ignore[typeddict-item]
        )
        result = run_tech_summary_agent(tech_state)["output"]

    elif tool_name == "evaluate_startup":
        eval_state = StartupEvalState(
            input={
                "startupId": startup["startupId"],
                "name":      startup["name"],
                "team":      startup["team"],
                "funding":   startup["funding"],
                "traction":  startup["traction"],
            },
            output={},  # type: ignore[typeddict-item]
        )
        result = run_startup_eval_agent(eval_state)["output"]

    else:
        result = {"error": f"알 수 없는 툴: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)


def _send_to_next_stage(state: SearchAgentState, analyses: dict) -> None:
    """
    다음 단계 에이전트에 state와 분석 결과를 전달합니다.

    TODO: 다음 단계 에이전트 구현 예정
    Args:
        state:    SearchAgentState (output.startupList 포함)
        analyses: 기업명 → {market_eval, tech_summary}
    """
    _ = state, analyses  # TODO: 다음 단계 에이전트 연결 시 제거


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

def run_search_agent(state: SearchAgentState) -> SearchAgentState:
    """
    국내 반도체 스타트업을 탐색하고 분석을 orchestrate합니다.

    Args:
        state: SearchAgentState — input.searchCriteria에 탐색 조건 설정

    Returns:
        state — output.startupList / fetchedAt / totalFetched 가 채워진 상태
        ※ 분석 결과(market_eval, tech_summary)는 _send_to_next_stage()로 전달됩니다.
    """
    client = OpenAI()
    criteria = state["input"]["searchCriteria"]

    # Step 1: 스타트업 목록 수집
    print(f"📋 스타트업 탐색 중 (도메인: {criteria.get('targetDomain')}) ...")
    startup_list: list[StartupInfo] = _get_startup_list(criteria)
    print(f"✅ {len(startup_list)}개 스타트업 수집 완료\n")

    # Step 2: Supervisor loop — 시장성 평가 + 기술 요약
    print("🤖 시장성 평가 및 기술 요약 시작...\n")

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 반도체 스타트업 분석을 총괄하는 에이전트입니다.\n\n"
                "주어진 스타트업 목록 전체에 대해 다음 작업을 수행하세요:\n"
                "1. 각 스타트업마다 evaluate_market 툴로 시장성 평가\n"
                "2. 각 스타트업마다 summarize_technology 툴로 기술 요약\n"
                "3. 각 스타트업마다 evaluate_startup 툴로 종합 평가\n"
                "4. 모든 기업의 세 툴 호출이 완료되면 작업 종료\n\n"
                f"총 {len(startup_list)}개 기업 모두에 대해 세 툴을 빠짐없이 호출해야 합니다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"다음 반도체 스타트업 {len(startup_list)}개를 분석해주세요:\n\n"
                f"{json.dumps(startup_list, ensure_ascii=False, indent=2)}\n\n"
                "각 스타트업에 대해 evaluate_market, summarize_technology, evaluate_startup을 모두 호출하세요."
            ),
        },
    ]

    analyses: dict[str, dict] = {}

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=_TOOLS,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                tool_input = json.loads(tool_call.function.arguments)
                startup_name = tool_input.get("startup", {}).get("name", "unknown")
                print(f"  [{tool_call.function.name}] {startup_name}")

                result_str = _route_tool(tool_call.function.name, tool_input)
                result = json.loads(result_str)

                if startup_name not in analyses:
                    analyses[startup_name] = {}
                if tool_call.function.name == "evaluate_market":
                    analyses[startup_name]["market_eval"] = result
                elif tool_call.function.name == "summarize_technology":
                    analyses[startup_name]["tech_summary"] = result
                elif tool_call.function.name == "evaluate_startup":
                    analyses[startup_name]["startup_eval"] = result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        elif choice.finish_reason == "stop":
            print(f"\n✅ 분석 완료 — {len(analyses)}개 기업")
            break

        else:
            print(f"⚠️  예상치 못한 finish_reason: {choice.finish_reason}")
            break

    # Step 3: state output 채우기
    state["output"] = SearchOutput(
        startupList=startup_list,
        fetchedAt=datetime.now(timezone.utc).isoformat(),
        totalFetched=len(startup_list),
        analyses=analyses,
    )

    # Step 4: 다음 단계 에이전트에 전달
    _send_to_next_stage(state, analyses)

    return state
