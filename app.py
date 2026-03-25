"""
app.py — Phase 5
Main Gradio frontend for the Neural Networks RAG Demo.
"""

import gradio as gr
from pathlib import Path

# Local imports
from chunker   import load_and_chunk_all
from embedder  import build_index
from retriever import retrieve
from generator import generate_answer

# ── 1. App Startup Sequence ───────────────────────────────────────────────────

print("[app] Starting initialization...")
DATA_DIR = Path(__file__).parent / "data"

# Phase 1: Load and chunk all 4 transcripts
chunks = load_and_chunk_all(DATA_DIR)

# Phase 2: Build FAISS index and metadata list
index, chunk_metadata = build_index(chunks)

print("[app] Initialization complete. Ready for queries.")

# ── 2. Gradio Interface Logic ────────────────────────────────────────────────

def on_submit(question: str) -> tuple[str, str, str, str, str]:
    """
    Handle a user question.
    Returns:
        answer, source_1, source_2, source_3, warning_text
    """
    if not question.strip():
        return ("Please enter a question.", "", "", "", "")

    # Retrieve top 3 chunks
    results = retrieve(question, index, chunk_metadata, top_k=3)
    if not results:
        return ("No relevant information found.", "", "", "", "")

    # Generate answer with Groq
    answer = generate_answer(question, results)

    # Format source blocks
    sources = []
    for i, r in enumerate(results, 1):
        text = (
            f"**Video:** {r['video_name']}\n"
            f"**Timestamp:** {r['start_time_formatted']}\n"
            f"**Confidence:** {r['confidence']} (Score: {r['score']:.2f})\n\n"
            f"_{r['text'][:250]}..._"
        )
        sources.append(text)

    # Pad sources list to exactly 3 items to match Gradio outputs
    while len(sources) < 3:
        sources.append("")

    # Check for low confidence warning (if top result is < 0.3)
    warning = ""
    if results[0]["confidence"] == "Low":
        warning = "⚠️ **Low confidence result** — please verify with the source video."

    return (answer, sources[0], sources[1], sources[2], warning)


# ── 3. Gradio UI Layout ──────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("# Neural Networks & Deep Learning — Video Q&A")
    gr.Markdown("Ask questions about 4 videos from 3Blue1Brown, CampusX and CodeWithHarry \n\n*A local RAG demonstration powered by FAISS and OpenRouter*")

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about neural networks or deep learning...",
                lines=2,
                placeholder="e.g. How does 3Blue1Brown explain the relationship between pixel brightness and a neuron's activation?"
            )
            submit_btn = gr.Button("Submit", variant="primary")
            warning_out = gr.Markdown("")

            # Pre-filled Example Questions
            gr.Examples(
                examples=[
                    "How does 3Blue1Brown explain the relationship between pixel brightness and a neuron's activation?",
                    "What example does CodeWithHarry use to explain supervised learning?",
                    "How does a next-token predictor become a chatbot?"
                ],
                inputs=question_input
            )

        with gr.Column(scale=3):
            answer_out = gr.Markdown("### Answer will appear here...")

    gr.Markdown("---")
    gr.Markdown("### Sources Retrieved")
    with gr.Row():
        source1_out = gr.Markdown("")
        source2_out = gr.Markdown("")
        source3_out = gr.Markdown("")

    # Event binding
    submit_btn.click(
        fn=on_submit,
        inputs=[question_input],
        outputs=[answer_out, source1_out, source2_out, source3_out, warning_out],
    )
    question_input.submit(
        fn=on_submit,
        inputs=[question_input],
        outputs=[answer_out, source1_out, source2_out, source3_out, warning_out],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
