"""Scrapes AMFI Investor Corner topic pages and saves as JSONL.

Output: data/raw/amfi/articles.jsonl

amfiindia.com/investor is a Next.js SPA — URLs extracted from RSC payload 2026-06-30.
Two base paths (verified live):
  Knowledge Centre  → /investor/knowledge-center-info?zoneName=
  Investor Centre   → /investor/become-mf-distributor?zoneName=
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.amfiindia.com"
_KC = f"{BASE_URL}/investor/knowledge-center-info?zoneName="   # Knowledge Centre
_IC = f"{BASE_URL}/investor/become-mf-distributor?zoneName="  # Investor Centre
REQUEST_DELAY = 1.5
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "amfi"

# ponytail: hardcoded — re-probe RSC payload if AMFI restructures
# (url, label, category)
TOPICS: list[tuple[str, str, str]] = [
    # --- Knowledge Centre ---
    (_KC + "HistoryOfMutualFundsInIndia",           "HistoryOfMutualFundsInIndia",          "mutual-fund-basics"),
    (_KC + "IntroductionMutualFunds",               "IntroductionMutualFunds",              "mutual-fund-basics"),
    (_KC + "TypesOfMutualFundSchemes",              "TypesOfMutualFundSchemes",             "mutual-fund-basics"),
    (_KC + "expenseRatio",                          "expenseRatio",                         "mutual-fund-basics"),
    (_KC + "riskInMutualFunds",                     "riskInMutualFunds",                    "mutual-fund-basics"),
    (_KC + "AdvantagesOfInvestingInMutualFunds",    "AdvantagesOfInvestingInMutualFunds",   "mutual-fund-basics"),
    (_KC + "CategorizationOfMutualFundSchemes",     "CategorizationOfMutualFundSchemes",    "mutual-fund-basics"),
    (_KC + "NetAssetValueNAV",                      "NetAssetValueNAV",                     "mutual-fund-basics"),
    (_KC + "MythsAndFactsAboutMutualFunds",         "MythsAndFactsAboutMutualFunds",        "mutual-fund-basics"),
    (_KC + "DirectPlan",                            "DirectPlan",                           "mutual-fund-basics"),
    (_KC + "CutOffTimingsAndNewRuleOnApplicableNAV","CutOffTimingsAndNewRuleOnApplicableNAV","mutual-fund-basics"),
    # --- Investor Centre ---
    (_IC + "HowtoInvestInMF",                      "HowtoInvestInMF",                      "mutual-fund-basics"),
    (_IC + "KnowYourCustomer",                      "KnowYourCustomer",                     "investor-protection"),
    (_IC + "HowTowithdrawMoneyInMF",                "HowTowithdrawMoneyInMF",               "mutual-fund-basics"),
    (_IC + "sip",                                   "sip",                                  "mutual-fund-basics"),
    (_IC + "MinorAttainingMajority",                "MinorAttainingMajority",               "investor-protection"),
    (_IC + "deathOfUnitHolder",                     "deathOfUnitHolder",                    "grievance-redressal"),
    (_IC + "nomination",                            "nomination",                           "investor-protection"),
    (_IC + "trackYourMF",                           "trackYourMF",                          "mutual-fund-basics"),
    (_IC + "downloadVariousForms",                  "downloadVariousForms",                 "investor-protection"),
    (_IC + "InvestorService",                       "InvestorService",                      "grievance-redressal"),
    (_IC + "changeBankDetails",                     "changeBankDetails",                    "investor-protection"),
    (_IC + "consolidatedAcct",                      "consolidatedAcct",                     "mutual-fund-basics"),
]


def get_topic_urls() -> list[tuple[str, str, str]]:
    """Return (url, label, category) for every live AMFI topic."""
    return TOPICS


def scrape_topic(url: str, label: str, category: str) -> dict | None:
    """Fetch one topic page and return a raw record, or None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  SKIP {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Pages use Material UI (SSR). The h2 is the section heading; its grandparent
    # div contains all Q&A content. The div.maxWidth sibling only holds a toolbar.
    title_el = soup.find("h2") or soup.find("h1")
    content_el = None
    if title_el and title_el.parent and title_el.parent.parent:
        content_el = title_el.parent.parent

    title = title_el.get_text(strip=True) if title_el else label
    text = content_el.get_text(separator="\n", strip=True) if content_el else ""

    if not text:
        print(f"  SKIP {url}: no content extracted")
        return None

    return {
        "source_domain": "amfi",
        "source_category": category,
        "source_url": url,
        "title": title,
        "text": text,
    }


def run(limit: int | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "articles.jsonl"

    topics = get_topic_urls()
    if limit:
        topics = topics[:limit]
    print(f"Scraping {len(topics)} AMFI topic URLs")

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, (url, label, category) in enumerate(topics):
            print(f"[{i+1}/{len(topics)}] {url}")
            record = scrape_topic(url, label, category)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            if i < len(topics) - 1:
                time.sleep(REQUEST_DELAY)

    print(f"Done. {written}/{len(topics)} records written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape AMFI Investor Corner")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run(limit=None if args.all else args.limit)
