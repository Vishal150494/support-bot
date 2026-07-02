"""Downloads SEBI investor-education FAQ PDFs from the SEBI FAQ directory page.

Output:
  data/raw/sebi/pdfs/          — downloaded PDFs
  data/raw/sebi/pdf_manifest.jsonl — {filename, source_url, source_category} per PDF

Run parse_sebi_pdfs.py next to extract text.
"""

import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

FAQ_DIRECTORY_URL = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doFaq=yes"
BASE_URL = "https://www.sebi.gov.in"
PDF_DIR = Path(__file__).parents[2] / "data" / "raw" / "sebi" / "pdfs"
MANIFEST_PATH = PDF_DIR.parent / "pdf_manifest.jsonl"
REQUEST_DELAY = 1.5

# Keywords to match against link text (lower-cased) -> our taxonomy (section 3 shortlist)
WANTED: list[tuple[str, str]] = [
    ("mutual fund", "mutual-fund-basics"),
    ("glossary", "investor-education"),
    ("investor grievance", "complaints"),
    ("scores", "complaints"),
    ("general", "investor-education"),
    ("kyc", "regulations"),
    ("investor education", "investor-education"),
    ("derivative", "regulations"),
    ("secondary market", "regulations"),
    ("buyback", "regulations"),
    ("buy back", "regulations"),
    ("portfolio manager", "regulations"),
    ("settlement", "regulations"),
]


def _match_category(link_text: str) -> str | None:
    lower = link_text.lower()
    for keyword, category in WANTED:
        if keyword in lower:
            return category
    return None


def discover_pdfs() -> list[tuple[str, str, str]]:
    """Return (pdf_url, filename, category) for each relevant PDF on the directory page."""
    resp = requests.get(FAQ_DIRECTORY_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    seen: set[str] = set()
    results: list[tuple[str, str, str]] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        category = _match_category(a.get_text(strip=True))
        if category is None:
            continue
        full_url = urljoin(BASE_URL, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        filename = href.rstrip("/").split("/")[-1]
        results.append((full_url, filename, category))

    return results


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"  FAIL {url}: {e}")
        return False


def run() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs()
    print(f"Found {len(pdfs)} relevant PDFs on SEBI FAQ directory")

    manifest_records: list[dict] = []
    for i, (url, filename, category) in enumerate(pdfs):
        print(f"[{i+1}/{len(pdfs)}] {filename}  ({category})")
        dest = PDF_DIR / filename
        ok = download_pdf(url, dest)
        if ok:
            manifest_records.append({"filename": filename, "source_url": url, "source_category": category})
        if i < len(pdfs) - 1:
            time.sleep(REQUEST_DELAY)

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Done. {len(manifest_records)} PDFs downloaded. Manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    run()
