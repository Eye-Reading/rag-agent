"""
시장성 평가 에이전트 (RAG 기반)

역할: 입력받은 기업의 시장성을 분석하여 리턴
실행: SearchAgent.output.startupList 각 항목을 input으로 받아 10개 동시 실행
임베딩: BAAI/bge-m3  |  벡터스토어: ChromaDB
"""
import json
from typing import TypedDict, Optional
from openai import OpenAI

from .rag.market_rag import MarketEvalRAG


# ──────────────────────────────────────────
# State 정의
# ──────────────────────────────────────────

class MarketEvalInput(TypedDict):
    startupId: str
    domain: str
    targetSegment: str   # 기업이 공략하는 세부 시장


class MarketEvalOutput(TypedDict):
    startupId: str
    marketScore: int     # 1~10
    scoringReason: str   # score 산출 근거
    sources: list[str]   # 정보 출처 URL 또는 문서명


class MarketEvalState(TypedDict):
    input: MarketEvalInput
    output: MarketEvalOutput  # TODO: 추후 state 정의 예정


# ──────────────────────────────────────────
# RAG 싱글톤
# ──────────────────────────────────────────

# TODO: 시장성 평가 기준을 아래에 직접 작성하세요
EVALUATION_CRITERIA = """
평가 기준:
- 기준 1:
- 기준 2:
- 기준 3:
- 기준 4:
- 기준 5:
"""

_rag: Optional[MarketEvalRAG] = None


def get_rag() -> MarketEvalRAG:
    global _rag
    if _rag is None:
        _rag = MarketEvalRAG()
    return _rag


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

def run_market_eval_agent(state: MarketEvalState) -> MarketEvalState:
    """
    입력받은 기업의 시장성을 RAG 기반으로 분석하여 state.output을 채워 반환합니다.

    Args:
        state: MarketEvalState — input.startupId / domain / targetSegment 설정 필요

    Returns:
        state — output이 채워진 상태
    """
    inp = state["input"]

    # Step 1: RAG — 관련 시장 데이터 검색
    rag = get_rag()
    query = f"{inp['domain']} {inp['targetSegment']} 반도체 시장 규모 성장률 동향"
    context_docs = rag.retrieve(query, n_results=5)

    if context_docs:
        rag_context = "참고 시장 데이터 (RAG 검색 결과):\n" + "\n---\n".join(context_docs)
    else:
        rag_context = "참고 시장 데이터: 없음 (벡터스토어가 비어 있어 일반 지식으로 평가합니다)"

    # Step 2: OpenAI — 검색된 컨텍스트 + 평가 기준으로 시장성 평가
    client = OpenAI()

    schema_example = json.dumps({
        "startupId": inp["startupId"],
        "tam": "120B USD",
        "targetSegment": inp["targetSegment"],
        "cagr": "23%",
        "marketScore": 7,
        "scoringReason": "점수 산출 근거 (2-3문장)",
        "sources": ["https://example.com/report", "Gartner 2024 반도체 시장 보고서"]
    }, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 반도체 시장 분석 전문가입니다. "
                    "제공된 시장 데이터(RAG 검색 결과)와 평가 기준을 바탕으로 "
                    "TAM, CAGR 등 시장성 지표를 객관적으로 분석합니다. "
                    "기업 평가 시 한국기업평가(https://www.korearatings.com/)의 "
                    "신용등급 및 기업 분석 데이터를 우선적으로 참고하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"다음 반도체 기업의 시장성을 평가해주세요.\n\n"
                    f"## 기업 정보\n"
                    f"- startupId: {inp['startupId']}\n"
                    f"- 도메인: {inp['domain']}\n"
                    f"- 타겟 세그먼트: {inp['targetSegment']}\n\n"
                    f"## {rag_context}\n\n"
                    f"## 평가 기준\n{EVALUATION_CRITERIA}\n\n"
                    f"## 응답 형식\n"
                    f"아래 JSON 스키마로만 응답하세요 (다른 텍스트 없이).\n"
                    f"marketScore는 1~10 정수로 산출하세요.\n"
                    f"{schema_example}"
                ),
            },
        ],
    )

    text = response.choices[0].message.content or ""
    start, end = text.find('{'), text.rfind('}') + 1

    if start == -1 or end == 0:
        output = MarketEvalOutput(
            startupId=inp["startupId"],
            marketScore=0,
            scoringReason="파싱 실패",
            sources=[],
        )
    else:
        try:
            parsed = json.loads(text[start:end])
            output = MarketEvalOutput(
                startupId=parsed["startupId"],
                marketScore=parsed["marketScore"],
                scoringReason=parsed["scoringReason"],
                sources=parsed.get("sources", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            output = MarketEvalOutput(
                startupId=inp["startupId"],
                marketScore=0,
                scoringReason="파싱 실패",
                sources=[],
            )

    state["output"] = output
    return state
