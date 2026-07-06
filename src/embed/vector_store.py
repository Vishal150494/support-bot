"""Builds and queries a ChromaDB vector store from pre-computed embeddings.

Build (ingestion):
  Loads embeddings.npy + chunks.jsonl (must be parallel / same row order),
  upserts everything into a persistent ChromaDB collection with cosine distance.

Query (retrieval):
  search_text(query, n_results) embeds the query with encode_query() and
  returns the top-n chunks with metadata and similarity scores.

Input:  data/processed/embeddings.npy   — float32 [n_chunks, 1024]
        data/processed/chunks.jsonl     — parallel metadata per row
Output: data/vectordb/                  — persistent ChromaDB on disk
"""

import json

import chromadb
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
CHUNKS_FILE   = PROCESSED_DIR / "chunks.jsonl"
EMBED_FILE    = PROCESSED_DIR / "embeddings.npy"
DB_DIR        = Path(__file__).parents[2] / "data" / "vectordb"

COLLECTION_NAME = "support-bot"
UPSERT_BATCH    = 500   # stay well under ChromaDB's internal batch limit


def _collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # embeddings are L2-normalised
    )


def build() -> None:
    """Load embeddings + chunks and upsert into the ChromaDB collection."""
    if not EMBED_FILE.exists():
        raise FileNotFoundError(f"Run embed.py (or Colab notebook) first: {EMBED_FILE}")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Run chunk.py first: {CHUNKS_FILE}")

    embeddings = np.load(EMBED_FILE)                                      # (N, 1024)
    records    = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]

    if len(embeddings) != len(records):
        raise ValueError(
            f"Row mismatch: {len(embeddings)} embeddings vs {len(records)} chunks. "
            "Re-run chunk.py then embed.py to realign."
        )

    print(f"Indexing {len(records)} chunks into '{COLLECTION_NAME}' …")
    col = _collection()

    for start in range(0, len(records), UPSERT_BATCH):
        batch_recs  = records[start : start + UPSERT_BATCH]
        batch_embs  = embeddings[start : start + UPSERT_BATCH]

        col.upsert(
            ids        = [r["chunk_id"] for r in batch_recs],
            embeddings = batch_embs.tolist(),
            documents  = [r["text"] for r in batch_recs],
            metadatas  = [
                {
                    "source_domain":   r["source_domain"],
                    "source_category": r["source_category"],
                    "source_url":      r["source_url"],
                    "title":           r["title"],
                }
                for r in batch_recs
            ],
        )
        print(f"  upserted {start + len(batch_recs)}/{len(records)}")

    print(f"Done. Collection '{COLLECTION_NAME}' has {col.count()} items -> {DB_DIR}")


def search_text(
    query: str,
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Embed query with encode_query() and return top-n chunks.

    Each result dict: {text, chunk_id, source_domain, source_category,
                       source_url, title, score}
    score is cosine similarity (1.0 = identical, 0.0 = orthogonal).

    'where' is an optional ChromaDB metadata filter, e.g.:
        {"source_domain": "zerodha"}
    """
    from sentence_transformers import SentenceTransformer  # lazy — skip on build-only runs

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    q_emb = model.encode_query(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    col = _collection()
    kwargs = dict(
        query_embeddings = q_emb.tolist(),
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)

    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append(
            {
                "text":            doc,
                "chunk_id":        meta.get("chunk_id", ""),
                "source_domain":   meta["source_domain"],
                "source_category": meta["source_category"],
                "source_url":      meta["source_url"],
                "title":           meta["title"],
                "score":           round(1 - dist, 4),  # cosine distance → similarity
            }
        )
    return results


def run() -> None:
    build()


if __name__ == "__main__":
    run()
