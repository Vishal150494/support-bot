"""Queries the ChromaDB vector store built by src/embed/vector_store.py.

search_text(query, n_results) embeds the query with encode_query() and
returns the top-n chunks with metadata and cosine similarity scores.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # financial text contains ₹ and other non-cp1252 chars

import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).parents[2] / "data" / "vectordb"  # fix: was "vectodb"

COLLECTION_NAME = "support-bot"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Module-level cache — avoids reloading the model on every search call
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_ID)
    return _model


def _collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # embeddings are L2-normalised
    )


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
    query_emb = _get_model().encode_query(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    col = _collection()
    kwargs = dict(
        query_embeddings=query_emb.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
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
                "source_domain":   meta["source_domain"],
                "source_category": meta["source_category"],  # fix: was "source_caegory"
                "source_url":      meta["source_url"],
                "title":           meta["title"],
                "score":           round(1 - dist, 4),  # cosine distance -> similarity
            }
        )
    return results


def run() -> None:
    results = search_text("what is NAV in mutual funds", n_results=3)
    for r in results:
        print(f"[{r['score']}] {r['title']} ({r['source_domain']})")
        print(f"  {r['text']} …\n")


if __name__ == "__main__":
    run()
