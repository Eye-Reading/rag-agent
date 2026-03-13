"""
투자 판단 에이전트

역할:
  1. searchCorp 에이전트에서 과락 필터를 통과한 기업 데이터를 수신
  2. Team(35%) · Market(25%) · Tech(25%) 항목별 비중 점수 계산
  3. Qdrant RAG 기반 성공 롤모델 DNA 유사도(15%) 점수 계산
  4. 총점(100점 만점) 내림차순 투자 우선순위 랭킹 생성
  5. 보고서 생성 에이전트로 State 전달
"""
import json
from typing import NotRequired, Optional, TypedDict

from .rag import DnaRoleModelRAG


# ──────────────────────────────────────────
# 공유 타입 (search_agent.py 의 StartupInfo · RejectionRecord 호환 재정의)
# 모듈 간 직접 의존을 피하기 위해 필요한 구조만 로컬에 선언합니다.
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


class PassedCompany(TypedDict):
    """
    searchCorp 에이전트가 투자 판단 에이전트로 전달하는 과락 통과 기업 데이터.
    search_agent.py 의 RejectionRecord 에 references 필드를 추가한 확장 구조입니다.
    _send_to_next_stage 연결 시 _build_record 함수에 references 필드 추가 필요.
    """
    companyName: str
    totalScore: int             # 과락 필터 원점수 합산 (marketScore + techScore + startupScore)
    marketScore: int            # 시장성 원점수 1~10
    marketScoringReason: str
    techScore: int              # 기술력 원점수 1~10
    techScoringReason: str
    startupScore: int           # 팀·투자·트랙션 종합 원점수 1~10
    startupScoringReason: str
    startupInfo: StartupInfo    # DNA 유사도 계산용 원본 기업 데이터
    references: list[str]       # market_eval · tech_summary · startup_eval 출처 합산


# ──────────────────────────────────────────
# 투자 판단 결과 타입
# ──────────────────────────────────────────

class WeightedItemScore(TypedDict):
    """단일 평가 항목의 비중 적용 결과"""
    rawScore: int           # 이전 에이전트 원점수 (1~10)
    weightedScore: float    # rawScore / 10 * 항목 만점 (소수점 2자리)
    maxScore: float         # 항목 만점: team=35.0, market=25.0, tech=25.0
    reason: str             # 이전 에이전트로부터 전달받은 점수 산출 근거


class DnaWeightedScore(TypedDict):
    """
    DNA 유사도 항목 점수 — Qdrant 벡터 검색으로 직접 산출.
    성공 롤모델(NVIDIA · Qualcomm · AMD)과의 코사인 유사도를 15점 만점으로 환산합니다.
    """
    weightedScore: float            # 0~15점
    maxScore: float                 # 15.0 (고정)
    similarCompanies: list[str]     # 유사도 상위 롤모델 기업명 목록
    similarityScores: list[float]   # 각 롤모델과의 코사인 유사도 값 (0.0~1.0)
    reason: str                     # 어떤 롤모델과 얼마나 유사했는지 서술


class InvestmentRanking(TypedDict):
    """투자 우선순위 랭킹 단위 — 보고서 생성 에이전트가 직접 소비합니다."""
    rank: int                           # 투자 우선순위 (1위 = 최고 총점)
    companyName: str
    totalScore: float                   # 합산 가중 점수 (0~100점)
    teamScore: WeightedItemScore        # 창업자 및 핵심 인력 (35%)
    marketScore: WeightedItemScore      # 시장성 (25%)
    techScore: WeightedItemScore        # 제품/기술력 (25%)
    dnaScore: DnaWeightedScore          # 성공 롤모델 DNA 유사도 (15%)
    references: list[str]               # 보고서 작성용 출처 (모든 에이전트 sources 합산)
    startupInfo: StartupInfo            # 원본 기업 데이터


# ──────────────────────────────────────────
# LangGraph State
# ──────────────────────────────────────────

class InvestmentDecisionInput(TypedDict):
    allRejected: bool
    passReport: list[PassedCompany]     # 과락 통과 기업 목록 (allRejected=False)
    rejectionReport: list[PassedCompany]  # 전원 과락 시 보고서 에이전트에 그대로 전달


class InvestmentDecisionOutput(TypedDict):
    allRejected: bool
    rankings: list[InvestmentRanking]       # 총점 내림차순 정렬 (allRejected=False)
    rejectionReport: list[PassedCompany]    # allRejected=True 일 때 다음 노드로 그대로 전달
    report: NotRequired[str]                # 보고서 작성 에이전트가 채우는 최종 마크다운 보고서


class InvestmentDecisionState(TypedDict):
    input: InvestmentDecisionInput
    output: InvestmentDecisionOutput


# ──────────────────────────────────────────
# DNA 유사도 계산 유틸 (Step 3)
# ──────────────────────────────────────────

TEAM_MAX_SCORE = 35.0
MARKET_MAX_SCORE = 25.0
TECH_MAX_SCORE = 25.0
DNA_MAX_SCORE = 15.0

_dna_rag: Optional[DnaRoleModelRAG] = None


def get_dna_rag() -> DnaRoleModelRAG:
    """DNA 롤모델 RAG 싱글톤을 반환합니다."""
    global _dna_rag
    if _dna_rag is None:
        _dna_rag = DnaRoleModelRAG()
    return _dna_rag


def _serialize_startup_info_for_dna(startup_info: StartupInfo) -> str:
    """
    searchCorp의 StartupInfo 관점을 유지해 DNA 비교용 텍스트를 생성합니다.
    동일한 직렬화 포맷을 사용하면 롤모델 벡터와 의미 비교가 안정적입니다.
    """
    payload = {
        "startupId": startup_info.get("startupId", ""),
        "name": startup_info.get("name", ""),
        "foundedYear": startup_info.get("foundedYear", "미공개"),
        "domain": startup_info.get("domain", "미공개"),
        "location": startup_info.get("location", "미공개"),
        "stage": startup_info.get("stage", "미공개"),
        "team": startup_info.get("team", {"founderCount": 0, "founderProfiles": []}),
        "funding": startup_info.get(
            "funding",
            {
                "totalFunding": "미공개",
                "latestRound": "미공개",
                "latestValuation": "미공개",
                "keyInvestors": [],
            },
        ),
        "traction": startup_info.get(
            "traction",
            {
                "revenueYear": [],
                "arrGrowthRate": "미공개",
                "keyCustomers": [],
            },
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def calculate_dna_weighted_score(startup_info: StartupInfo) -> DnaWeightedScore:
    """
    단일 스타트업의 DNA 유사도 점수를 계산합니다.

    계산식:
        dna_score = average_similarity * 15.0

    - average_similarity: 상위 롤모델 유사도 평균 (0.0 ~ 1.0)
    - DNA 만점: 15점
    """
    rag = get_dna_rag()
    query_text = _serialize_startup_info_for_dna(startup_info)
    points = rag.search_similar(query_text=query_text, top_k=3)

    if not points:
        return DnaWeightedScore(
            weightedScore=0.0,
            maxScore=DNA_MAX_SCORE,
            similarCompanies=[],
            similarityScores=[],
            reason="롤모델 벡터 데이터가 없어 DNA 유사도 점수를 0점으로 처리했습니다.",
        )

    similar_companies = [str(p.payload.get("company", "unknown")) for p in points if p.payload]
    similarity_scores = [round(float(p.score), 4) for p in points]

    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    weighted_score = round(avg_similarity * DNA_MAX_SCORE, 2)

    top_name = similar_companies[0] if similar_companies else "unknown"
    top_score = similarity_scores[0] if similarity_scores else 0.0

    return DnaWeightedScore(
        weightedScore=weighted_score,
        maxScore=DNA_MAX_SCORE,
        similarCompanies=similar_companies,
        similarityScores=similarity_scores,
        reason=(
            f"상위 롤모델 {', '.join(similar_companies)}와의 코사인 유사도 평균 "
            f"{avg_similarity:.4f}를 15점 만점으로 환산했습니다. "
            f"가장 유사한 기업은 {top_name}({top_score:.4f})입니다."
        ),
    )


def _clamp_raw_score(score: int) -> int:
    """원점수 범위를 0~10으로 보정합니다."""
    return max(0, min(10, int(score)))


def _calculate_weighted_item_score(
    raw_score: int,
    max_score: float,
    reason: str,
) -> WeightedItemScore:
    normalized = _clamp_raw_score(raw_score)
    weighted = round((normalized / 10.0) * max_score, 2)
    return WeightedItemScore(
        rawScore=normalized,
        weightedScore=weighted,
        maxScore=max_score,
        reason=reason,
    )


def _build_investment_ranking(company: PassedCompany) -> InvestmentRanking:
    """과락 통과 기업 1건에 대한 최종 투자 점수와 랭킹 단위를 구성합니다."""
    team_score = _calculate_weighted_item_score(
        raw_score=company.get("startupScore", 0),
        max_score=TEAM_MAX_SCORE,
        reason=company.get("startupScoringReason", ""),
    )
    market_score = _calculate_weighted_item_score(
        raw_score=company.get("marketScore", 0),
        max_score=MARKET_MAX_SCORE,
        reason=company.get("marketScoringReason", ""),
    )
    tech_score = _calculate_weighted_item_score(
        raw_score=company.get("techScore", 0),
        max_score=TECH_MAX_SCORE,
        reason=company.get("techScoringReason", ""),
    )

    startup_info = company.get("startupInfo", {})
    dna_score = calculate_dna_weighted_score(startup_info)

    total_score = round(
        team_score["weightedScore"]
        + market_score["weightedScore"]
        + tech_score["weightedScore"]
        + dna_score["weightedScore"],
        2,
    )

    return InvestmentRanking(
        rank=0,  # 정렬 이후 재할당
        companyName=company.get("companyName", "unknown"),
        totalScore=total_score,
        teamScore=team_score,
        marketScore=market_score,
        techScore=tech_score,
        dnaScore=dna_score,
        references=company.get("references", []),
        startupInfo=startup_info,
    )


def _send_to_next_stage(state: InvestmentDecisionState) -> None:
    """
    보고서 작성 LangGraph 워크플로우를 호출하고,
    생성된 마크다운 보고서를 state["output"]["report"]에 저장합니다.

    Args:
        state: InvestmentDecisionState — output 이 채워진 상태
    """
    import os  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    # investDecision 패키지 기준으로 main 디렉토리를 sys.path에 추가합니다.
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    from reportWriter.graph import report_graph  # pylint: disable=import-outside-toplevel

    output = state["output"]
    pipeline_input = {
        "allRejected": output.get("allRejected", False),
        "rankings": output.get("rankings", []),
        "rejectionReport": output.get("rejectionReport", []),
        "report": "",
    }

    result = report_graph.invoke(pipeline_input)
    output["report"] = result.get("report", "")  # type: ignore[typeddict-unknown-key]
    print(f"\n[InvestmentDecision] 보고서 작성 완료 — {len(output['report'])}자")


def _log_investment_output(output: InvestmentDecisionOutput) -> None:
    """투자 판단 결과를 운영 로그로 출력합니다."""
    if output.get("allRejected", False):
        rejected_count = len(output.get("rejectionReport", []))
        print(f"\n📤 [InvestmentDecision] 전원 과락 — 보고서 노드로 반려 데이터 {rejected_count}건 전달")
        return

    rankings = output.get("rankings", [])
    print(f"\n📤 [InvestmentDecision] 보고서 노드 전달 데이터 — 랭킹 {len(rankings)}건")
    for ranking in rankings:
        company = ranking.get("companyName", "unknown")
        rank = ranking.get("rank", 0)
        total = ranking.get("totalScore", 0.0)

        team = ranking.get("teamScore", {}).get("weightedScore", 0.0)
        market = ranking.get("marketScore", {}).get("weightedScore", 0.0)
        tech = ranking.get("techScore", {}).get("weightedScore", 0.0)
        dna = ranking.get("dnaScore", {}).get("weightedScore", 0.0)

        dna_reason = ranking.get("dnaScore", {}).get("reason", "")
        refs = ranking.get("references", [])

        print(
            f"  {rank}위 {company} | 총점 {total}점 "
            f"(Team {team} + Market {market} + Tech {tech} + DNA {dna})"
        )
        print(f"     - DNA 근거: {dna_reason}")
        print(f"     - 참고자료 {len(refs)}건")


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

def run_investment_decision_agent(state: InvestmentDecisionState) -> InvestmentDecisionState:
    """
    투자 판단 에이전트 메인 노드.

    Args:
        state: InvestmentDecisionState
            - input.allRejected    : 전원 과락 여부
            - input.passReport     : 과락 통과 기업 리스트 (각 항목 원점수 + 코멘트 + 출처 + 원본 데이터)
            - input.rejectionReport: 전원 과락 시 보고서 에이전트로 넘길 반려 데이터

    Returns:
        state — output 이 채워진 상태
            - output.allRejected   : 그대로 전달
            - output.rankings      : 총점 내림차순 InvestmentRanking 리스트 (통과 기업이 있을 때)
            - output.rejectionReport: 전원 과락 시 그대로 전달
    """
    inp = state["input"]
    all_rejected = bool(inp.get("allRejected", False))

    if all_rejected:
        state["output"] = InvestmentDecisionOutput(
            allRejected=True,
            rankings=[],
            rejectionReport=inp.get("rejectionReport", []),
        )
        _log_investment_output(state["output"])
        _send_to_next_stage(state)
        return state

    pass_report = inp.get("passReport", [])
    rankings = [_build_investment_ranking(company) for company in pass_report]

    # 총점 내림차순 정렬 후 1위부터 랭크 부여
    rankings.sort(key=lambda x: x["totalScore"], reverse=True)
    for idx, ranking in enumerate(rankings, start=1):
        ranking["rank"] = idx

    state["output"] = InvestmentDecisionOutput(
        allRejected=False,
        rankings=rankings,
        rejectionReport=[],
    )
    _log_investment_output(state["output"])
    _send_to_next_stage(state)
    return state
