"""
투자 판단 에이전트

역할:
  1. searchCorp 에이전트에서 과락 필터를 통과한 기업 데이터를 수신
  2. Team(35%) · Market(25%) · Tech(25%) 항목별 비중 점수 계산
  3. Qdrant RAG 기반 성공 롤모델 DNA 유사도(15%) 점수 계산
  4. 총점(100점 만점) 내림차순 투자 우선순위 랭킹 생성
  5. 보고서 생성 에이전트로 State 전달
"""
from typing import TypedDict


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
    성공 롤모델(NVIDIA · TSMC · SK하이닉스)과의 코사인 유사도를 15점 만점으로 환산합니다.
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


class InvestmentDecisionState(TypedDict):
    input: InvestmentDecisionInput
    output: InvestmentDecisionOutput


# ──────────────────────────────────────────
# 진입점 (구현은 Step 4에서 완성)
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
    # TODO: Step 4에서 구현
    raise NotImplementedError("Step 4에서 구현 예정")
