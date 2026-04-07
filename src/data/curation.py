from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.request import urlopen


def _strip_html(text: str) -> str:
    no_scripts = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    no_styles = re.sub(r"<style.*?</style>", " ", no_scripts, flags=re.S | re.I)
    no_tags = re.sub(r"<[^>]+>", " ", no_styles)
    normalized = re.sub(r"\s+", " ", unescape(no_tags)).strip()
    return normalized


def scrape_documents(urls: Iterable[str], timeout: int = 20) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for url in urls:
        with urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        text = _strip_html(raw)
        if text:
            docs.append({"source": url, "text": text})
    return docs


def save_jsonl_documents(documents: Iterable[Dict[str, Any]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_jsonl_documents(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    docs: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs
