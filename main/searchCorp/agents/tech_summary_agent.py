"""
기술 요약 에이전트

역할: 입력받은 기업의 핵심 기술을 분석하여 리턴
실행: SearchAgent.output.startupList 각 항목을 input으로 받아 10개 동시 실행
"""
import json
from typing import TypedDict
from openai import OpenAI


# ──────────────────────────────────────────
# State 정의
# ──────────────────────────────────────────

class TechSummaryInput(TypedDict):
    startupId: str
    name: str
    domain: str


class TechSummaryOutput(TypedDict):
    startupId: str
    techScore: int       # 1~10
    scoringReason: str   # 점수 산출 근거
    sources: list[str]   # 정보 출처 URL 또는 문서명


class TechSummaryState(TypedDict):
    input: TechSummaryInput
    output: TechSummaryOutput


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

def run_tech_summary_agent(state: TechSummaryState) -> TechSummaryState:
    """
    입력받은 기업의 핵심 기술을 분석하여 state.output을 채워 반환합니다.

    Args:
        state: TechSummaryState — input.startupId / name / domain 설정 필요

    Returns:
        state — output이 채워진 상태
    """
    inp = state["input"]
    client = OpenAI()

    schema_example = json.dumps({
        "startupId": inp["startupId"],
        "products": [{
            "name": "제품명",
            "launchedYear": "2023",
            "target": "데이터센터",
            "foundryProcess": "5nm",
            "status": "출시완료"
        }],
        "patentCount": 12,
        "tapedOutCount": 3,
        "techDifferentiation": "경쟁사 대비 기술 차별점 설명 (3-4문장)",
        "foundryPartnership": ["TSMC", "삼성파운드리"],
        "techScore": 7,
        "scoringReason": "제품 포트폴리오 완성도, 특허 수 및 질적 수준, 테이프아웃 경험, 파운드리 파트너십 신뢰도, 경쟁사 대비 기술 차별점을 각각 분석한 상세 근거 (5문장 이상)",
        "sources": ["https://example.com/news", "특허청 공개특허 2024-XXXXX"]
    }, ensure_ascii=False, indent=2)

    system_msg = {
        "role": "system",
        "content": (
            "당신은 반도체 기술 전문가입니다. "
            "웹 검색을 통해 스타트업의 제품 포트폴리오, 특허, 테이프아웃 이력, "
            "파운드리 파트너십을 정확하게 분석합니다. "
            "불확실한 수치는 0 또는 '미공개'로 표기하세요. "
            "기업 평가 시 한국기업평가(https://www.korearatings.com/)의 "
            "신용등급 및 기업 분석 데이터를 우선적으로 참고하세요."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"다음 반도체 스타트업의 기술을 분석해주세요.\n\n"
            f"## 기업 정보\n"
            f"- startupId: {inp['startupId']}\n"
            f"- 기업명: {inp['name']}\n"
            f"- 도메인: {inp['domain']}\n\n"
            f"먼저 웹에서 해당 기업의 최신 기술 정보를 검색하세요. "
            f"검색 후 아래 JSON 스키마로만 최종 응답하세요 (다른 텍스트 없이).\n"
            f"techScore는 1~10 정수로 산출하세요.\n"
            f"{schema_example}"
        ),
    }

    # Step 1: 웹 검색 포함 응답
    search_response = client.chat.completions.create(
        model="gpt-4o-search-preview",
        messages=[system_msg, user_msg],
    )
    search_content = search_response.choices[0].message.content or ""

    # Step 2: 검색 결과를 바탕으로 JSON 추출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            system_msg,
            user_msg,
            {"role": "assistant", "content": search_content},
            {
                "role": "user",
                "content": (
                    f"위 분석을 바탕으로 아래 JSON 스키마로만 응답하세요 (다른 텍스트 없이):\n"
                    f"{schema_example}"
                ),
            },
        ],
    )

    text = response.choices[0].message.content or ""
    start, end = text.find('{'), text.rfind('}') + 1

    if start == -1 or end == 0:
        output = TechSummaryOutput(
            startupId=inp["startupId"],
            techScore=0,
            scoringReason="파싱 실패",
            sources=[],
        )
    else:
        try:
            parsed = json.loads(text[start:end])
            output = TechSummaryOutput(
                startupId=parsed["startupId"],
                techScore=parsed["techScore"],
                scoringReason=parsed["scoringReason"],
                sources=parsed.get("sources", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            output = TechSummaryOutput(
                startupId=inp["startupId"],
                techScore=0,
                scoringReason="파싱 실패",
                sources=[],
            )

    state["output"] = output
    return state
