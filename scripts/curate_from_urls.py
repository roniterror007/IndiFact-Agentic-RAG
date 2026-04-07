from __future__ import annotations

import argparse
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace_root))

from src.data.curation import save_jsonl_documents, scrape_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape URL list into JSONL corpus")
    parser.add_argument("--out", default="datasets/curated_corpus.jsonl")
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args()

    docs = scrape_documents(args.urls)
    save_jsonl_documents(docs, args.out)
    print(f"Saved {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
