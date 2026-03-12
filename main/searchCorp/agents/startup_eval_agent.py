"""
기업 종합 평가 에이전트

역할: startupList의 팀·투자·트랙션 정보를 바탕으로 기업 종합 점수를 산출
실행: SearchAgent.output.startupList 각 항목을 input으로 받아 10개 동시 실행
"""
import json
from typing import TypedDict
from openai import OpenAI


# ──────────────────────────────────────────
# State 정의
# ──────────────────────────────────────────

class StartupEvalInput(TypedDict):
    startupId: str
    name: str
    team: dict      # founderCount, founderProfiles[]
    funding: dict   # totalFunding, latestRound, latestValuation, keyInvestors
    traction: dict  # revenueYear, arrGrowthRate, keyCustomers


class StartupEvalOutput(TypedDict):
    startupId: str
    finalScore: int     # 1~10
    scoringReason: str  # 점수 산출 근거
    sources: list[str]  # 정보 출처 URL 또는 문서명


class StartupEvalState(TypedDict):
    input: StartupEvalInput
    output: StartupEvalOutput


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

def run_startup_eval_agent(state: StartupEvalState) -> StartupEvalState:
    """
    팀·투자·트랙션 정보를 종합하여 기업 최종 점수를 산출합니다.

    Args:
        state: StartupEvalState — input.startupId / name / team / funding / traction 설정 필요

    Returns:
        state — output이 채워진 상태
    """
    inp = state["input"]
    client = OpenAI()

    schema_example = json.dumps({
        "startupId": inp["startupId"],
        "teamScore": 7,
        "fundingScore": 6,
        "tractionScore": 8,
        "finalScore": 7,
        "scoringReason": "팀·투자·트랙션 종합 평가 근거 (2-3문장)",
        "sources": ["https://www.korearatings.com/", "https://example.com/news"]
    }, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 스타트업 투자 심사 전문가입니다. "
                    "팀 역량, 투자 현황, 트랙션을 종합적으로 평가하여 최종 점수를 산출합니다. "
                    "기업 평가 시 한국기업평가(https://www.korearatings.com/)의 "
                    "신용등급 및 기업 분석 데이터를 우선적으로 참고하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"다음 스타트업을 종합 평가해주세요.\n\n"
                    f"## 기업 정보\n"
                    f"- startupId: {inp['startupId']}\n"
                    f"- 기업명: {inp['name']}\n\n"
                    f"## 팀\n{json.dumps(inp['team'], ensure_ascii=False, indent=2)}\n\n"
                    f"## 투자 현황\n{json.dumps(inp['funding'], ensure_ascii=False, indent=2)}\n\n"
                    f"## 트랙션\n{json.dumps(inp['traction'], ensure_ascii=False, indent=2)}\n\n"
                    f"## 평가 기준\n"
                    f"- 팀 역량: 창업자 도메인 경력, 학력, 이전 재직 회사, 엑싯 경험\n"
                    f"- 투자 현황: 누적 투자금, 투자 단계, 주요 투자자 신뢰도\n"
                    f"- 트랙션: 매출 성장률, 주요 고객사 규모\n\n"
                    f"## 응답 형식\n"
                    f"아래 JSON 스키마로만 응답하세요 (다른 텍스트 없이).\n"
                    f"각 항목 점수(1~10)를 기반으로 finalScore(1~10)를 산출하세요.\n"
                    f"{schema_example}"
                ),
            },
        ],
    )

    text = response.choices[0].message.content or ""
    start, end = text.find('{'), text.rfind('}') + 1

    if start == -1 or end == 0:
        output = StartupEvalOutput(
            startupId=inp["startupId"],
            finalScore=0,
            scoringReason="파싱 실패",
            sources=[],
        )
    else:
        try:
            parsed = json.loads(text[start:end])
            output = StartupEvalOutput(
                startupId=parsed["startupId"],
                finalScore=parsed["finalScore"],
                scoringReason=parsed["scoringReason"],
                sources=parsed.get("sources", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            output = StartupEvalOutput(
                startupId=inp["startupId"],
                finalScore=0,
                scoringReason="파싱 실패",
                sources=[],
            )

    state["output"] = output
    return state
