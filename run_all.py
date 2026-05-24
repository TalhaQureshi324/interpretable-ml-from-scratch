"""
run_all.py
==========
Master runner for ML Assignment 2.
Executes all experiments for Q1 (k-Means), Q2 (Gaussian Naive Bayes),
and Q3 (C4.5 Decision Tree), generating all plots and summaries.
"""

import sys
import json
import os
from datetime import datetime

from q1_kmeans import run_q1
from q2_naive_bayes import run_q2
from q3_decision_tree import run_q3


def main():
    print("=" * 80)
    print("MACHINE LEARNING ASSIGNMENT 2 — FULL EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    results = {}

    # Run Q1
    try:
        results["q1"] = run_q1()
    except Exception as e:
        print(f"\n[Q1 ERROR] {e}", file=sys.stderr)
        results["q1"] = {"error": str(e)}

    # Run Q2
    try:
        results["q2"] = run_q2()
    except Exception as e:
        print(f"\n[Q2 ERROR] {e}", file=sys.stderr)
        results["q2"] = {"error": str(e)}

    # Run Q3
    try:
        results["q3"] = run_q3()
    except Exception as e:
        print(f"\n[Q3 ERROR] {e}", file=sys.stderr)
        results["q3"] = {"error": str(e)}

    # Save summary JSON
    os.makedirs("results", exist_ok=True)
    summary_path = "results/summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] Summary -> {summary_path}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
