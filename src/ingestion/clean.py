"""Cleans raw scraped records: strips residual HTML, deduplicates, normalises whitespace.

Input:  data/raw/zerodha/articles.jsonl
        data/raw/amfi/articles.jsonl
        data/raw/sebi/parsed/sebi_parsed.jsonl
Output: data/processed/cleaned.jsonl
"""

import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "processed"

RAW_FILES: list[Path] = [
    RAW_DIR / "zerodha" / "articles.jsonl",
    RAW_DIR / "amfi" / "articles.jsonl",
    RAW_DIR / "sebi" / "parsed" / "sebi_parsed.jsonl",
]

MIN_TEXT_LENGTH = 50  # drop near-empty articles


def _strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(separator="\n")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_record(record: dict) -> dict | None:
    text = _normalize(_strip_html(record.get("text", "")))
    if len(text) < MIN_TEXT_LENGTH:
        return None
    return {**record, "text": text}


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cleaned.jsonl"

    seen_fingerprints: set[str] = set()
    written = skipped_short = skipped_dupe = 0

    with output_path.open("w", encoding="utf-8") as out:
        for raw_file in RAW_FILES:
            if not raw_file.exists():
                print(f"  SKIP (missing): {raw_file}")
                continue
            print(f"Cleaning {raw_file.name} ...")
            with raw_file.open(encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    cleaned = clean_record(record)
                    if cleaned is None:
                        skipped_short += 1
                        continue
                    # Deduplicate on leading 200 chars of text
                    fp = cleaned["text"][:200]
                    if fp in seen_fingerprints:
                        skipped_dupe += 1
                        continue
                    seen_fingerprints.add(fp)
                    out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    written += 1

    print(
        f"Done. {written} written, {skipped_short} too-short, {skipped_dupe} dupes → {output_path}"
    )


if __name__ == "__main__":
    run()
