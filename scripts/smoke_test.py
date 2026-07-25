"""End-to-end smoke test: run the canonical queries through the agent and
verify the planted synthetic patterns are detected.

Run offline (no LLM):  GEMINI_API_KEY= python scripts/smoke_test.py
Run with Gemini:       python scripts/smoke_test.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset, clean_dataset
from src.agent.orchestrator import run_agent

QUERIES = [
    "Find structuring patterns in the last 30 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is account 90000100 suspicious?",
    "Analyse this dataset for suspicious activity",
]

# accounts planted by scripts/generate_synthetic.py
STRUCTURING_ACCOUNTS = {str(90_000_100 + i) for i in range(6)}


def main():
    df = clean_dataset(load_dataset())
    failures = []

    for q in QUERIES:
        print(f"\n=== {q}")
        result = run_agent(q, df)
        print(f"    parser: {result.get('parser_used')} | intent: {result.get('detected_intent')}")
        print(f"    tools:  {' -> '.join(result.get('tools_invoked', []))}")

        if result.get("error"):
            print(f"    ERROR: {result['error']}")
            failures.append((q, result["error"]))
            continue

        if result.get("aggregation_result") is not None:
            n = len(result["aggregation_result"])
            print(f"    aggregation rows: {n}")
            if n == 0:
                failures.append((q, "aggregation returned no rows"))
        if result.get("flags"):
            flagged_accounts = {str(f["Sender_account"]) for f in result["flags"]}
            print(f"    flags: {result['flag_count']} | high risk: {result['high_risk_count']}")
            for f in result["flags"][:3]:
                print(f"      {f['Sender_account']} [{f['risk_level']}] {f['explanation'][:90]}")
            if "structuring" in q.lower():
                hits = flagged_accounts & STRUCTURING_ACCOUNTS
                print(f"    planted structuring accounts in top flags: {len(hits)}/{len(flagged_accounts)}")
                if not hits:
                    failures.append((q, "no planted structuring account in top flags"))
        if result.get("eda_summary"):
            print(f"    eda rows profiled: {result['eda_summary'].get('row_count')}")

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILURES ({len(failures)}):")
        for q, err in failures:
            print(f"  - {q}: {err}")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
