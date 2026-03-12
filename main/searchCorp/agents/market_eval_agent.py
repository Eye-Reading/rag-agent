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
    output: MarketEvalOutput


# ──────────────────────────────────────────
# RAG 싱글톤
# ──────────────────────────────────────────

# 운영 환경 기준에 맞춰 항목을 보완해 사용할 수 있습니다.
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

    # Step 2: OpenAI — RAG 컨텍스트 + 웹서치 + 평가 기준으로 시장성 평가
    client = OpenAI()

    schema_example = json.dumps({
        "startupId": inp["startupId"],
        "tam": "120B USD",
        "targetSegment": inp["targetSegment"],
        "cagr": "23%",
        "marketScore": 7,
        "scoringReason": "TAM 규모 및 신뢰성, CAGR 수준, 타겟 세그먼트의 경쟁 강도, 시장 진입 가능성을 각각 분석한 상세 근거 (5문장 이상)",
        "sources": ["https://example.com/report", "Gartner 2024 반도체 시장 보고서"]
    }, ensure_ascii=False, indent=2)

    system_msg = {
        "role": "system",
        "content": (
            "당신은 반도체 시장 분석 전문가입니다. "
            "웹 검색을 통해 최신 시장 데이터를 수집하고, "
            "TAM, CAGR 등 시장성 지표를 객관적으로 분석합니다. "
            "기업 평가 시 한국기업평가(https://www.korearatings.com/)의 "
            "신용등급 및 기업 분석 데이터를 우선적으로 참고하세요."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"다음 반도체 기업의 시장성을 평가해주세요.\n\n"
            f"## 기업 정보\n"
            f"- startupId: {inp['startupId']}\n"
            f"- 도메인: {inp['domain']}\n"
            f"- 타겟 세그먼트: {inp['targetSegment']}\n\n"
            f"## {rag_context}\n\n"
            f"## 평가 기준\n{EVALUATION_CRITERIA}\n\n"
            f"먼저 웹에서 해당 기업과 시장에 대한 최신 정보를 검색하세요. "
            f"검색 후 아래 JSON 스키마로만 최종 응답하세요 (다른 텍스트 없이).\n"
            f"marketScore는 1~10 정수로 산출하세요.\n"
            f"{schema_example}"
        ),
    }

    # Step 3: 웹 검색 — RAG 컨텍스트 + 최신 인터넷 데이터 수집
    search_response = client.chat.completions.create(
        model="gpt-4o-search-preview",
        messages=[system_msg, user_msg],
    )
    search_content = search_response.choices[0].message.content or ""

    # Step 4: RAG + 웹서치 결과 기반 JSON 추출
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
