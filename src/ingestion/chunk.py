"""Splits cleaned documents into overlapping chunks with per-chunk metadata.

Recursive structural splitting: paragraph > line > sentence > word.
Tables and Q&A pairs are kept whole where possible; oversized atoms get a
targeted secondary split (table rows batched with header repeat, Q&A answer
split at paragraphs with question prefix).

Input:  data/processed/cleaned.jsonl
Output: data/processed/chunks.jsonl
  {chunk_id, text, source_domain, source_category, source_url, title}
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

CLEANED_FILE = Path(__file__).parents[2] / "data" / "processed" / "cleaned.jsonl"
OUTPUT_FILE = Path(__file__).parents[2] / "data" / "processed" / "chunks.jsonl"

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
# Coarse-to-fine separator hierarchy
SEPARATORS = ["\n\n", "\n", ". ", " "]

TABLE_ROW_BATCH = 40   # rows per sub-table chunk (header repeated in each)
QA_PREFIX_CHARS = 240  # question chars to prepend to each split answer piece


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:40]


def _protect_atoms(text: str) -> tuple[str, dict[str, str]]:
    """Replace table blocks and Q&A pairs with single-line placeholder tokens.

    Placeholders use null-byte delimiters so no separator can split within them.
    """
    lines = text.split("\n")
    out: list[str] = []
    atoms: dict[str, str] = {}
    i = 0

    def _save(block: list[str], tag: str) -> str:
        key = f"\x00{tag}{len(atoms)}\x00"
        atoms[key] = "\n".join(block)
        return key

    while i < len(lines):
        line = lines[i]

        # Table block: consecutive lines containing " | " (pdfplumber + our format)
        if " | " in line:
            block = []
            while i < len(lines) and " | " in lines[i]:
                block.append(lines[i])
                i += 1
            out.append(_save(block, "T"))
            continue

        # Q&A pair: question (ends with ?) + answer lines until blank or next question
        if line.rstrip().endswith("?") and i + 1 < len(lines) and lines[i + 1].strip():
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].rstrip().endswith("?"):
                block.append(lines[i])
                i += 1
            out.append(_save(block, "Q"))
            continue

        out.append(line)
        i += 1

    return "\n".join(out), atoms


def _recursive_split(text: str, seps: list[str], max_tok: int) -> list[str]:
    """Split at the first separator level that yields sub-limit pieces.

    Greedy left-to-right merge keeps adjacent small pieces packed together.
    If no separator fits and text still exceeds max_tok, keep it whole
    (natural unit too large to split — invariant from user spec).
    """
    if _tokens(text) <= max_tok or not seps:
        return [text] if text.strip() else []

    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = (buf + sep + piece) if buf else piece
        if _tokens(candidate) <= max_tok:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if _tokens(piece) > max_tok:
                chunks.extend(_recursive_split(piece, rest, max_tok))
                buf = ""
            else:
                buf = piece
    if buf:
        chunks.append(buf)

    return [c.strip() for c in chunks if c.strip()]


def _resplit_oversized(chunk: str) -> list[str]:
    """Secondary pass: break chunks that survived atom-restoration still over-limit.

    Three targeted rules applied in order:
      1. Pure table  → split every TABLE_ROW_BATCH rows, repeat header
      2. Q&A block   → split answer at paragraph breaks, prefix each with question
      3. Plain text  → fall back to word-level recursive split
    """
    if _tokens(chunk) <= CHUNK_TOKENS:
        return [chunk]

    lines = [l for l in chunk.split("\n") if l.strip()]

    # Rule 1: pure table (every content line contains " | ")
    if lines and all(" | " in l for l in lines):
        header = lines[0]
        rows = lines[1:]
        return [
            (header + "\n" + "\n".join(rows[i : i + TABLE_ROW_BATCH])).strip()
            for i in range(0, max(1, len(rows)), TABLE_ROW_BATCH)
        ]

    # Rule 2: Q&A block (first line is the question)
    if lines and lines[0].rstrip().endswith("?"):
        question_prefix = lines[0][:QA_PREFIX_CHARS]
        answer = "\n".join(lines[1:])
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return [f"{question_prefix}\n{p}" for p in paragraphs]

    # Rule 3: plain oversized text — word-level split, no atom protection
    return _recursive_split(chunk, SEPARATORS, CHUNK_TOKENS)


def split_into_chunks(text: str) -> list[str]:
    protected, atoms = _protect_atoms(text)
    raw = _recursive_split(protected, SEPARATORS, CHUNK_TOKENS)

    def _restore(s: str) -> str:
        for k, v in atoms.items():
            s = s.replace(k, v)
        return s

    restored = [piece for c in raw for piece in _resplit_oversized(_restore(c))]

    # Prepend tail of previous chunk as overlap context
    overlap_words = max(1, int(OVERLAP_TOKENS * 0.75))
    result: list[str] = []
    for i, chunk in enumerate(restored):
        if i > 0:
            tail = " ".join(restored[i - 1].split()[-overlap_words:])
            chunk = (tail + " " + chunk) if tail else chunk
        result.append(chunk)
    return result


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
            url_hash = hashlib.md5(record["source_url"].encode()).hexdigest()[:6]
            chunks = split_into_chunks(record["text"])

            for idx, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{domain}_{title_slug}_{url_hash}_{idx:03d}",
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
