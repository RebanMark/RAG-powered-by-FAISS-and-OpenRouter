"""
retriever.py — Phase 3
Searches FAISS for the top-3 chunks most relevant to a user query.
"""

import faiss
import numpy as np
from embedder import embed_query


def _confidence_label(score: float) -> str:
    if score > 0.5:
        return "High"
    elif score >= 0.3:
        return "Medium"
    return "Low"


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunk_metadata: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Embed the query, search FAISS, and return the top-k results with
    confidence scores attached.

    Each result dict contains:
        text, video_name, start_time, start_time_formatted,
        score (float), confidence (str: High / Medium / Low)
    """
    query_vec = embed_query(query)                        # shape: (1, dim)
    scores, indices = index.search(query_vec, top_k)     # scores shape: (1, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:          # FAISS returns -1 when fewer vectors than top_k
            continue
        meta = chunk_metadata[idx].copy()
        meta["score"]      = float(score)
        meta["confidence"] = _confidence_label(float(score))
        results.append(meta)

    return results


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from chunker import load_and_chunk_all
    from embedder import build_index

    chunks = load_and_chunk_all(Path(__file__).parent / "data")
    index, meta = build_index(chunks)

    test_query = "How does 3Blue1Brown explain pixel brightness and neuron activation?"
    print(f"\nQuery: {test_query!r}\n")

    results = retrieve(test_query, index, meta)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['video_name']} @ {r['start_time_formatted']}")
        print(f"     Score: {r['score']:.4f}  Confidence: {r['confidence']}")
        print(f"     Text preview: {r['text'][:120]}…\n")
