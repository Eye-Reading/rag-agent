"""
보고서 작성 에이전트

역할:
  1. 투자 판단 에이전트에서 전달받은 State(allRejected, rankings, rejectionReport)를 수신
  2. allRejected 플래그에 따라 Case A(투자 추천 순위) 또는 Case B(전원 보류) 분기
  3. 분기에 맞는 System Prompt와 데이터를 매핑하여 LLM 호출 준비
  4. LLM이 생성한 최종 마크다운 보고서를 State에 저장
"""
import json
from datetime import date
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .prompts.case_a import CASE_A_SYSTEM_PROMPT
from .prompts.case_b import CASE_B_SYSTEM_PROMPT

# 보고서 품질 우선 — gpt-4o 사용. 필요 시 환경변수나 호출 측에서 교체 가능합니다.
_REPORT_MODEL = "gpt-4o"
_MAX_TOKENS = 16000  # 전문 투자 보고서 분량 확보

_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    """보고서 작성용 LLM 싱글톤을 반환합니다."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=_REPORT_MODEL, max_tokens=_MAX_TOKENS)
    return _llm


# ──────────────────────────────────────────
# State 정의
# ──────────────────────────────────────────

class ReportWriterInput(TypedDict):
    """투자 판단 에이전트의 InvestmentDecisionOutput을 그대로 수신합니다."""
    allRejected: bool
    rankings: list[dict]        # list[InvestmentRanking] — Case A
    rejectionReport: list[dict] # list[PassedCompany]     — Case B


class ReportWriterOutput(TypedDict):
    report: str  # LLM이 생성한 최종 마크다운 보고서


class ReportWriterState(TypedDict):
    input: ReportWriterInput
    output: ReportWriterOutput


# ──────────────────────────────────────────
# 전처리: 분기 판단 및 프롬프트·데이터 매핑
# ──────────────────────────────────────────

class _PromptContext(TypedDict):
    """LLM 호출에 필요한 완성된 프롬프트와 메타 정보."""
    system_prompt: str  # 데이터가 주입된 최종 System Prompt
    case: str           # "A" 또는 "B" — 로깅·디버깅용


def _serialize(data: list[dict]) -> str:
    """State 데이터를 LLM이 읽기 쉬운 들여쓰기 JSON 문자열로 직렬화합니다."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def _generate_report(prompt_ctx: _PromptContext) -> str:
    """
    준비된 프롬프트 컨텍스트를 LLM에 전달하여 마크다운 보고서를 생성합니다.

    System Prompt에 데이터가 이미 주입되어 있으므로,
    Human 메시지는 보고서 작성 트리거 역할만 합니다.

    Args:
        prompt_ctx: prepare_prompt_context()가 반환한 _PromptContext

    Returns:
        LLM이 생성한 마크다운 보고서 문자열
    """
    llm = _get_llm()
    case_label = "투자 추천 순위" if prompt_ctx["case"] == "A" else "전원 보류"
    print(f"\n[ReportWriter] LLM 호출 시작 — Case {prompt_ctx['case']} ({case_label} 보고서)")

    messages = [
        SystemMessage(content=prompt_ctx["system_prompt"]),
        HumanMessage(content="위 데이터를 바탕으로 지침에 따라 보고서를 작성해 주세요."),
    ]

    response = llm.invoke(messages)
    report: str = str(response.content)

    print(f"[ReportWriter] 보고서 생성 완료 — {len(report)}자")
    return report


def prepare_prompt_context(state: ReportWriterState) -> _PromptContext:
    """
    allRejected 플래그를 확인하여 Case A / Case B 프롬프트와 데이터를 매핑합니다.

    Case A (allRejected=False):
        - CASE_A_SYSTEM_PROMPT에 rankings 데이터와 report_date를 주입합니다.
    Case B (allRejected=True):
        - CASE_B_SYSTEM_PROMPT에 rejectionReport 데이터와 report_date를 주입합니다.

    Args:
        state: ReportWriterState

    Returns:
        _PromptContext — system_prompt(완성된 프롬프트 문자열), case("A"/"B")
    """
    inp = state["input"]
    all_rejected: bool = bool(inp.get("allRejected", False))
    report_date: str = date.today().isoformat()  # 예: "2026-03-12"

    if all_rejected:
        rejection_report = inp.get("rejectionReport", [])
        # totalScore 내림차순 정렬 — 과락 기업 간 상대적 비교를 위해
        sorted_data = sorted(rejection_report, key=lambda x: x.get("totalScore", 0), reverse=True)
        system_prompt = CASE_B_SYSTEM_PROMPT.format(
            rejection_json=_serialize(sorted_data),
            report_date=report_date,
        )
        return _PromptContext(system_prompt=system_prompt, case="B")

    rankings = inp.get("rankings", [])
    # 투자 판단 에이전트가 이미 rank 기준 정렬을 보장하지만, 방어적으로 재정렬합니다.
    sorted_data = sorted(rankings, key=lambda x: x.get("rank", 0))
    system_prompt = CASE_A_SYSTEM_PROMPT.format(
        rankings_json=_serialize(sorted_data),
        report_date=report_date,
    )
    return _PromptContext(system_prompt=system_prompt, case="A")


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

def run_report_writer_agent(state: ReportWriterState) -> ReportWriterState:
    """
    보고서 작성 에이전트 메인 노드.

    Args:
        state: ReportWriterState
            - input.allRejected    : 전원 과락 여부 (Case A/B 분기 기준)
            - input.rankings       : 투자 우선순위 랭킹 리스트 (Case A)
            - input.rejectionReport: 전원 과락 기업 상세 데이터 (Case B)

    Returns:
        state — output.report 에 마크다운 보고서가 저장된 상태
    """
    prompt_ctx = prepare_prompt_context(state)
    report = _generate_report(prompt_ctx)
    state["output"] = ReportWriterOutput(report=report)
    return state
