"""
반도체 스타트업 분석 시스템 진입점

실행 방법:
  cd main/searchCorp
  ANTHROPIC_API_KEY=<your-key> OPENAI_API_KEY=<your-key> python main.py

실행 순서:
  1. searchCorp  — 반도체 스타트업 탐색 및 시장성·기술·종합 평가
  2. investDecision — 가중치 합산 및 DNA 유사도 기반 투자 우선순위 랭킹
  3. reportWriter — LLM 기반 마크다운 투자 보고서 생성 → report_<날짜>.md 저장
"""
import sys
import os
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env 로드 — import 전에 실행해야 API 키가 환경변수에 등록됩니다.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# searchCorp 및 main 디렉토리를 sys.path에 추가합니다.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(_BASE_DIR)
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, _MAIN_DIR)

from agents.search_agent import run_search_agent, SearchAgentState, SearchCriteria


def main():
    print("=" * 60)
    print("🔬 국내 반도체 스타트업 분석 시스템")
    print("=" * 60)
    print()

    # 탐색 조건 설정
    initial_state: SearchAgentState = {
        "input": {
            "searchCriteria": SearchCriteria(
                targetDomain="AI반도체",
                targetStage="",         # 전체 단계
                targetRegion="",        # 전체 지역
                fetchCount=10,
                excludeList=[],
            )
        },
        "output": {
            "startupList": [],
            "fetchedAt": "",
            "totalFetched": 0,
            "analyses": {},
            "evaluations": [],
            "passed": [],
            "rejected": [],
            "allRejected": False,
            "passReport": [],
            "rejectionReport": [],
        }
    }

    result_state = run_search_agent(initial_state)

    output = result_state["output"]
    analyses = output.get("analyses", {})
    passed = output.get("passed", [])
    rejected = output.get("rejected", [])
    investment_decision = output.get("investmentDecision", {})

    print("\n" + "=" * 60)
    print(f"📊 수집 완료: {output['totalFetched']}개 기업 | {output['fetchedAt']}")
    print(f"   통과: {len(passed)}개  |  반려: {len(rejected)}개")
    print("=" * 60)

    print("\n✅ 통과 기업")
    for startup in output["startupList"]:
        name = startup["name"]
        if name not in passed:
            continue
        a = analyses.get(name, {})
        market_score = a.get("market_eval",  {}).get("marketScore", "-")
        tech_score   = a.get("tech_summary", {}).get("techScore",   "-")
        final_score  = a.get("startup_eval", {}).get("finalScore",  "-")
        total = sum(s for s in [market_score, tech_score, final_score] if isinstance(s, int))
        print(f"  • {name} ({startup.get('stage', '?')}) | 시장 {market_score} + 기술 {tech_score} + 종합 {final_score} = {total}점")

    print("\n❌ 반려 기업")
    for startup in output["startupList"]:
        name = startup["name"]
        if name not in rejected:
            continue
        a = analyses.get(name, {})
        market_score = a.get("market_eval",  {}).get("marketScore", "-")
        tech_score   = a.get("tech_summary", {}).get("techScore",   "-")
        final_score  = a.get("startup_eval", {}).get("finalScore",  "-")
        total = sum(s for s in [market_score, tech_score, final_score] if isinstance(s, int))
        print(f"  • {name} ({startup.get('stage', '?')}) | 시장 {market_score} + 기술 {tech_score} + 종합 {final_score} = {total}점")

    # JSON 결과 저장
    output_path = os.path.join(_BASE_DIR, "analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=2)

    investment_output_path = os.path.join(_BASE_DIR, "investment_decision_results.json")
    with open(investment_output_path, "w", encoding="utf-8") as f:
        json.dump(investment_decision, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_path}")
    print(f"투자판단 저장: {investment_output_path}")
    # PDF 보고서는 보고서 생성 에이전트 내부에서 main/searchCorp/ 에 저장됩니다.


if __name__ == "__main__":
    main()
