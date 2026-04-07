from __future__ import annotations

import json
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace_root))

from src.evaluation.harness import run_evaluation_harness


if __name__ == "__main__":
    result = run_evaluation_harness()
    print(json.dumps(result["summary"], indent=2))
    print("Saved detailed report to datasets/eval_results.json")
