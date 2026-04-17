from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace_root))

from src.evaluation.harness import run_evaluation_harness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline vs graph evaluation harness.")
    parser.add_argument("--corpus", default="datasets/sample_corpus.jsonl", help="Path to corpus JSONL")
    parser.add_argument("--claims", default="datasets/sample_eval.jsonl", help="Path to claims JSONL")
    parser.add_argument("--out", default="datasets/eval_results.json", help="Output path for result JSON")
    parser.add_argument(
        "--require-ragas",
        action="store_true",
        help="Exit with failure if ragas metrics are not available.",
    )
    args = parser.parse_args()

    result = run_evaluation_harness(
        corpus_path=args.corpus,
        claims_path=args.claims,
        out_path=args.out,
    )

    if args.require_ragas:
        ragas = result.get("ragas", {})
        enabled = bool(ragas.get("enabled", False))
        scores = ragas.get("scores", {}) if isinstance(ragas, dict) else {}
        score_values = list(scores.values()) if isinstance(scores, dict) else []
        has_valid_scores = bool(score_values) and all(v is not None for v in score_values)

        if not enabled or not has_valid_scores:
            print("Ragas is required but unavailable or produced invalid scores.")
            print(json.dumps(ragas, indent=2))
            raise SystemExit(1)

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved detailed report to {args.out}")
