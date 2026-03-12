"""
기술 요약 에이전트

역할: 입력받은 기업의 핵심 기술을 분석하여 리턴
실행: SearchAgent.output.startupList 각 항목을 input으로 받아 10개 동시 실행
"""
import json
from typing import TypedDict, Optional
import anthropic


# ──────────────────────────────────────────
# State 정의
# ──────────────────────────────────────────

class ProductInfo(TypedDict):
    name: str
    launchedYear: str
    target: str          # 타겟 시장 (예: "데이터센터", "온디바이스", "자동차")
    foundryProcess: str  # 제조 공정 (예: "5nm", "28nm", "미공개")
    status: str          # "출시완료" | "양산중" | "개발중"


class TechSummaryInput(TypedDict):
    startupId: str
    name: str
    domain: str


class TechSummaryOutput(TypedDict):
    startupId: str
    products: list[ProductInfo]
    patentCount: int            # 특허 개수
    tapedOutCount: int          # 실제 칩 생산(테이프아웃) 경험 횟수
    techDifferentiation: str    # 경쟁사 대비 기술 차별점
    foundryPartnership: list[str]  # 칩 제조 위탁 파트너사
    techScore: int              # 1~10
    scoringReason: str          # 점수 산출 근거


class TechSummaryState(TypedDict):
    input: TechSummaryInput
    output: TechSummaryOutput  # TODO: 추후 state 정의 예정


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
    client = anthropic.Anthropic()

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
        "scoringReason": "기술 점수 산출 근거 (2-3문장)"
    }, ensure_ascii=False, indent=2)

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=(
            "당신은 반도체 기술 전문가입니다. "
            "스타트업의 제품 포트폴리오, 특허, 테이프아웃 이력, 파운드리 파트너십을 "
            "정확하게 분석합니다. 불확실한 수치는 0 또는 '미공개'로 표기하세요."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"다음 반도체 스타트업의 기술을 분석해주세요.\n\n"
                f"## 기업 정보\n"
                f"- startupId: {inp['startupId']}\n"
                f"- 기업명: {inp['name']}\n"
                f"- 도메인: {inp['domain']}\n\n"
                f"## 응답 형식\n"
                f"아래 JSON 스키마로만 응답하세요 (다른 텍스트 없이).\n"
                f"techScore는 1~10 정수로 산출하세요.\n"
                f"{schema_example}"
            )
        }]
    ) as stream:
        final_message = stream.get_final_message()

    text = next(b.text for b in final_message.content if b.type == "text")
    start, end = text.find('{'), text.rfind('}') + 1

    if start == -1 or end == 0:
        output = TechSummaryOutput(
            startupId=inp["startupId"],
            products=[],
            patentCount=0,
            tapedOutCount=0,
            techDifferentiation=text,
            foundryPartnership=[],
            techScore=0,
            scoringReason="파싱 실패",
        )
    else:
        try:
            parsed = json.loads(text[start:end])
            output = TechSummaryOutput(**parsed)
        except (json.JSONDecodeError, TypeError):
            output = TechSummaryOutput(
                startupId=inp["startupId"],
                products=[],
                patentCount=0,
                tapedOutCount=0,
                techDifferentiation=text,
                foundryPartnership=[],
                techScore=0,
                scoringReason="파싱 실패",
            )

    state["output"] = output
    return state
