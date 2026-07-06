"""Embeds chunked documents using Qwen3-Embedding-0.6B and saves to disk.

Qwen3-Embedding is asymmetric — use the right method for each side:
  encode_document()  ingestion (this file) — no instruction prefix
  encode_query()     query time (retrieval) — adds task instruction prefix

Refer notebooks/embed_colab.ipynb to run this same script on google colab with T4 GPU.

Input:  data/processed/chunks.jsonl  (produced by chunk.py)
Output: data/processed/embeddings.npy  — float32 [n_chunks, dim], L2-normalised
        chunks.jsonl acts as the parallel metadata index (same row order)
"""
import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
EMBED_FILE = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 8
MAX_SEQ_LEN = 2048 # ~8000 chars; truncates oversized SEBI PDF chunks

def run() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Run chunk.py first: {CHUNKS_FILE}")
    
    texts = [json.loads(line)["text"] for line in CHUNKS_FILE.open(encoding="utf-8")]
    print(f"Embedding {len(texts)} chunks with {MODEL_ID}")

    # Warn if any chunk will be truncated
    long = sum(1 for t in texts if len(t) // 4 > MAX_SEQ_LEN)
    if long:
        print(f"  ⚠  {long} chunk(s) exceed {MAX_SEQ_LEN} tokens and will be truncated")

    model = SentenceTransformer(MODEL_ID)
    embeddings = model.encode_document(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    np.save(EMBED_FILE, embeddings.astype("float32"))
    print(f"Done. Shape: {embeddings.shape} -> {EMBED_FILE} ")


if __name__ == "__main__":
    run()
