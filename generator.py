"""
generator.py — Phase 4
Calls Groq (llama3-8b-8192) with retrieved chunks to generate a grounded answer.
"""

import os
from cerebras.cloud.sdk import Cerebras

MODEL         = "llama3.1-8b"
MAX_CHUNK_WORDS = 400


def _truncate(text: str, max_words: int = MAX_CHUNK_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _build_prompt(question: str, results: list[dict]) -> str:
    context_blocks = []
    for i, r in enumerate(results, 1):
        block = (
            f"[CHUNK {i} - Source: {r['video_name']} at {r['start_time_formatted']}]\n"
            f"{_truncate(r['text'])}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above:"
    )


def generate_answer(question: str, results: list[dict]) -> str:
    """
    Call Cerebras with the question + top-3 retrieved chunks.
    Returns the generated answer string.
    """
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        return "⚠️ CEREBRAS_API_KEY is not set. Please add it as an environment variable or Hugging Face Space Secret."

    client = Cerebras(api_key=api_key)

    system_msg = (
        "You are a helpful assistant that answers questions about neural networks "
        "and deep learning based strictly on video transcript content. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say "
        "\"I couldn't find a clear answer in the video transcripts.\" "
        "Do not use your own knowledge."
    )

    user_msg = _build_prompt(question, results)

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        model=MODEL,
        max_completion_tokens=1024,
        temperature=0.2,
        top_p=1,
        stream=False
    )

    return completion.choices[0].message.content.strip()


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from chunker   import load_and_chunk_all
    from embedder  import build_index
    from retriever import retrieve

    chunks = load_and_chunk_all(Path(__file__).parent / "data")
    index, meta = build_index(chunks)

    q = "How does 3Blue1Brown explain the relationship between pixel brightness and neuron activation?"
    results = retrieve(q, index, meta)

    print(f"Question: {q}\n")
    answer = generate_answer(q, results)
    print(f"Answer:\n{answer}")
