"""Splits cleaned documents into overlapping chunks with per-chunk metadata.

Target: ~300-500 tokens per chunk, ~50 token overlap (section 8 locked decisions).
Token estimate: 1 token ≈ 4 chars (good enough for English/financial text).

Input:  data/processed/cleaned.jsonl
Output: data/processed/chunks.jsonl  — one chunk record per line, matching section 3 schema:
  {chunk_id, text, source_domain, source_category, source_url, title}
"""

import json
import re
import unicodedata
from pathlib import Path

CLEANED_FILE = Path(__file__).parents[2] / "data" / "processed" / "cleaned.jsonl"
OUTPUT_FILE = Path(__file__).parents[2] / "data" / "processed" / "chunks.jsonl"

# Section 8 locked decisions
CHUNK_TOKENS = 400        # target tokens per chunk
OVERLAP_TOKENS = 50       # token overlap between adjacent chunks

# Convert token budgets to word counts (1 token ≈ 0.75 words)
_CHUNK_WORDS = max(1, int(CHUNK_TOKENS * 0.75))
_OVERLAP_WORDS = max(1, int(OVERLAP_TOKENS * 0.75))


def _slug(text: str) -> str:
    """ASCII slug for use in chunk_id, max 40 chars."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:40]


def split_text(text: str) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + _CHUNK_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - _OVERLAP_WORDS
    return chunks


def run() -> None:
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(
            f"Run clean.py first — cleaned file not found: {CLEANED_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with CLEANED_FILE.open(encoding="utf-8") as f_in, OUTPUT_FILE.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            title_slug = _slug(record.get("title", "untitled"))
            domain = record.get("source_domain", "unknown")
            chunks = split_text(record["text"])

            for idx, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{domain}_{title_slug}_{idx:03d}",
                    "text": chunk_text,
                    "source_domain": record["source_domain"],
                    "source_category": record["source_category"],
                    "source_url": record["source_url"],
                    "title": record["title"],
                }
                f_out.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Done. {written} chunks written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
