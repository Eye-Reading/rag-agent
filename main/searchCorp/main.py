"""
반도체 스타트업 분석 시스템 진입점

실행 방법:
  cd main/searchCorp
  ANTHROPIC_API_KEY=<your-key> python main.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    # 결과 저장
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
