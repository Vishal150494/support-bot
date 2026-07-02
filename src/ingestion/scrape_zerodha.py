"""Scrapes Zerodha Support Center articles via sitemap and saves as JSONL.

Output: data/raw/zerodha/articles.jsonl
Each line: {source_domain, source_category, source_url, title, text}
"""

import json
import re
import time
from pathlib import Path
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://support.zerodha.com/sitemap.xml"
EXCLUDE_PATH = "/contact-us"
REQUEST_DELAY = 1.5  # seconds — be a polite crawler
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "zerodha"

# Capture top-level category (group 1) and immediate subcategory (group 2).
# URLs: /category/{top}/{subcat}/[{subsub}/]articles/{slug}
ARTICLE_PATTERN = re.compile(
    r"support\.zerodha\.com/category/([^/]+)/([^/]+)/(?:[^/]+/)*articles/([^?#]+)"
)

# Subcategory checked first (more specific); top-level is the fallback.
# Both live in the same dict — no overlap between top-level and subcat slugs.
CATEGORY_MAP: dict[str, str] = {
    # top-level fallbacks
    "account-opening": "account-opening",
    "your-zerodha-account": "account-management",
    "trading-and-markets": "orders-execution",
    "funds": "funds-settlement",
    "mutual-funds": "coin-mutual-funds",
    "console": "general",
    # subcategory overrides (verified from sitemap 2026-06-30)
    "corporate-actions": "corporate-actions",
    "margins": "margins-leverage",
    "coin-general": "coin-mutual-funds",
    "features-on-coin": "coin-mutual-funds",
    "payments-and-orders": "coin-mutual-funds",
    "understanding-mutual-funds": "coin-mutual-funds",
    "adding-bank-accounts": "funds-settlement",
    "adding-funds": "funds-settlement",
    "fund-withdrawal": "funds-settlement",
    "transfer-of-shares-and-conversion-of-shares": "account-management",
    "account-modification-and-segment-addition": "account-management",
}


def fetch_article_urls(limit: int | None = None) -> list[tuple[str, str, str]]:
    """Return (url, top_cat, subcat) tuples sampled evenly across top-level categories."""
    resp = requests.get(SITEMAP_URL, timeout=15)
    resp.raise_for_status()

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ElementTree.fromstring(resp.content)

    from collections import defaultdict
    by_cat: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for loc in root.findall(".//sm:loc", ns):
        url = loc.text.strip()
        if EXCLUDE_PATH in url:
            continue
        m = ARTICLE_PATTERN.search(url)
        if m:
            by_cat[m.group(1)].append((url, m.group(1), m.group(2)))

    if not limit:
        return [item for items in by_cat.values() for item in items]

    per_cat = max(1, limit // len(by_cat))
    results: list[tuple[str, str, str]] = []
    for items in by_cat.values():
        results.extend(items[:per_cat])
    return results[:limit]


def scrape_article(url: str, top_cat: str, subcat: str) -> dict | None:
    """Fetch one article page and return a raw record, or None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  SKIP {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Zerodha support center uses Tailwind — h1 is always present; body is the
    # first classless <div> sibling immediately after the <h1>.
    title_el = soup.find("h1")
    body_el = None
    if title_el:
        # First sibling div with no class = article body (not nav/related/CTA sections)
        for sib in title_el.find_next_siblings("div"):
            if not sib.get("class"):
                body_el = sib
                break

    if not title_el or not body_el:
        print(f"  SKIP {url}: couldn't locate title/body elements")
        return None

    category = CATEGORY_MAP.get(subcat) or CATEGORY_MAP.get(top_cat, "general")
    return {
        "source_domain": "zerodha",
        "source_category": category,
        "source_url": url,
        "title": title_el.get_text(strip=True),
        "text": body_el.get_text(separator="\n", strip=True),
    }


def run(limit: int | None = 10) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "articles.jsonl"

    urls = fetch_article_urls(limit=limit)
    print(f"Fetched sitemap: {len(urls)} article URLs (limit={limit})")

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, (url, top_cat, subcat) in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] {url}")
            record = scrape_article(url, top_cat, subcat)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            if i < len(urls) - 1:
                time.sleep(REQUEST_DELAY)

    print(f"Done. {written}/{len(urls)} articles written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Zerodha Support Center")
    parser.add_argument("--limit", type=int, default=10, help="Articles to scrape (default 10 for smoke test)")
    parser.add_argument("--all", action="store_true", help="Scrape all articles (overrides --limit)")
    args = parser.parse_args()
    run(limit=None if args.all else args.limit)
