"""Extracts text from downloaded SEBI FAQ PDFs and saves as JSONL.

Input:  data/raw/sebi/pdfs/          — PDFs from scrape_sebi.py
        data/raw/sebi/pdf_manifest.jsonl
Output: data/raw/sebi/parsed/sebi_parsed.jsonl
"""

import json
from pathlib import Path

import pdfplumber

PDF_DIR = Path(__file__).parents[2] / "data" / "raw" / "sebi" / "pdfs"
MANIFEST_PATH = PDF_DIR.parent / "pdf_manifest.jsonl"
OUTPUT_DIR = PDF_DIR.parent / "parsed"


def extract_text(pdf_path: Path) -> str:
    """Return all page text + table content joined by double newline."""
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts: list[str] = []
            text = page.extract_text()
            if text and text.strip():
                parts.append(text.strip())
            for table in page.extract_tables():
                rows = [
                    " | ".join(str(cell or "").strip() for cell in row)
                    for row in table
                    if any(cell for cell in row)
                ]
                if rows:
                    parts.append("\n".join(rows))
            if parts:
                pages.append("\n\n".join(parts))
    return "\n\n".join(pages)


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("-", " ").replace("_", " ").title()


def run() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found — run scrape_sebi.py first: {MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sebi_parsed.jsonl"

    manifest: list[dict] = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            manifest.append(json.loads(line))

    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for entry in manifest:
            pdf_path = PDF_DIR / entry["filename"]
            if not pdf_path.exists():
                print(f"  SKIP (missing): {entry['filename']}")
                continue
            print(f"  Parsing: {entry['filename']}")
            text = extract_text(pdf_path)
            if not text.strip():
                print(f"  SKIP (no text extracted): {entry['filename']}")
                continue
            record = {
                "source_domain": "sebi",
                "source_category": entry["source_category"],
                "source_url": entry["source_url"],
                "title": _title_from_filename(entry["filename"]),
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. {written}/{len(manifest)} PDFs parsed -> {output_path}")


if __name__ == "__main__":
    run()
