"""
embedder.py — Phase 2
Embeds chunks using paraphrase-multilingual-MiniLM-L12-v2 and builds a FAISS index.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Module-level singletons — loaded once, reused across calls
_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_chunk_metadata: list[dict] = []


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model with retry logic for HF Spaces timeouts."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME}")
        import time
        for attempt in range(5):
            try:
                _model = SentenceTransformer(MODEL_NAME)
                break
            except Exception as e:
                print(f"[embedder] Model download attempt {attempt+1} failed: {e}")
                time.sleep(3)
        if _model is None:
            raise RuntimeError("Failed to load SentenceTransformer after multiple attempts.")
        print("[embedder] Model loaded.")
    return _model


def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Embed all chunks, normalise vectors, build FAISS IndexFlatIP.

    Returns:
        index          — FAISS index (inner product / cosine similarity)
        chunk_metadata — parallel list of chunk dicts (mirrors FAISS positions)
    """
    global _index, _chunk_metadata

    model = get_model()

    texts = [chunk["text"] for chunk in chunks]
    print(f"[embedder] Embedding {len(texts)} chunks …")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalise so inner-product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Metadata list mirrors FAISS positions exactly
    chunk_metadata = [
        {
            "text":                c["text"],
            "video_name":          c["video_name"],
            "start_time":          c["start_time"],
            "start_time_formatted": c["start_time_formatted"],
        }
        for c in chunks
    ]

    _index          = index
    _chunk_metadata = chunk_metadata

    print(f"[embedder] FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index, chunk_metadata


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string and return a normalised float32 vector."""
    model = get_model()
    vec   = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)
    return vec


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    from chunker import load_and_chunk_all
    from pathlib import Path

    chunks = load_and_chunk_all(Path(__file__).parent / "data")
    index, meta = build_index(chunks)
    print(f"\nIndex ready: {index.ntotal} entries")
    print(f"Sample metadata: {meta[0]['video_name']} @ {meta[0]['start_time_formatted']}")
