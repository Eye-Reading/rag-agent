"""
보고서 작성 LangGraph Workflow

노드 구성:
  report_writer — 투자 판단 에이전트 출력을 받아 마크다운 보고서를 생성합니다.

State 설계:
  LangGraph StateGraph는 단일 평탄 dict를 기반으로 동작합니다.
  투자 판단 에이전트의 InvestmentDecisionOutput 필드를 그대로 수신하고,
  보고서 생성 결과(report)를 추가하여 반환합니다.
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .agents.report_writer_agent import (
    ReportWriterOutput,
    ReportWriterState,
    run_report_writer_agent,
)


# ──────────────────────────────────────────
# Graph State
# ──────────────────────────────────────────

class ReportPipelineState(TypedDict):
    """
    보고서 작성 워크플로우의 공유 상태.

    - allRejected / rankings / rejectionReport: 투자 판단 에이전트 출력을 그대로 수신
    - report: 보고서 작성 노드가 채우는 최종 마크다운 보고서
    """
    allRejected: bool
    rankings: list[dict]        # list[InvestmentRanking] — Case A
    rejectionReport: list[dict] # list[PassedCompany]     — Case B
    report: str


# ──────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────

def report_writer_node(state: ReportPipelineState) -> ReportPipelineState:
    """
    LangGraph 노드 래퍼.
    평탄한 ReportPipelineState를 ReportWriterState 형태로 변환하여
    run_report_writer_agent를 호출하고, 결과를 다시 평탄한 state에 병합합니다.
    """
    writer_state = ReportWriterState(
        input={
            "allRejected": state["allRejected"],
            "rankings": state.get("rankings", []),
            "rejectionReport": state.get("rejectionReport", []),
        },
        output=ReportWriterOutput(report=""),
    )

    result = run_report_writer_agent(writer_state)
    state["report"] = result["output"]["report"]
    return state


# ──────────────────────────────────────────
# Graph 빌드 및 컴파일
# ──────────────────────────────────────────

def build_report_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """
    보고서 작성 StateGraph를 생성하고 컴파일하여 반환합니다.

    그래프 구조:
        START → report_writer → END
    """
    graph = StateGraph(ReportPipelineState)
    graph.add_node("report_writer", report_writer_node)
    graph.set_entry_point("report_writer")
    graph.add_edge("report_writer", END)
    return graph.compile()


# 모듈 임포트 시점에 컴파일하여 재사용합니다.
report_graph = build_report_graph()
