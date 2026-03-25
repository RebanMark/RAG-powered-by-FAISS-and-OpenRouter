---
title: Neural Networks & Deep Learning Q&A
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🧠 Neural Networks & Deep Learning — Video RAG Q&A

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.10-orange.svg)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/Spaces-HuggingFace-yellow.svg)](https://huggingface.co/spaces)

A **Retrieval-Augmented Generation (RAG)** application built to intelligently answer questions about Deep Learning, Neural Networks, and Transformers based strictly on transcript data from educational YouTube videos.

An open-source demonstration of building a practical, localized RAG pipeline.

---

## 📖 Overview

This project implements an end-to-end local RAG pipeline without heavy vector databases. It grounds a Large Language Model (`llama3.1-8b` via Cerebras Cloud) exclusively in the provided video transcripts. If the specific answer is not in the videos, the model will gracefully state that it cannot find the answer, avoiding hallucination.

### 🎥 Indexed Videos
1. **3Blue1Brown** — *But what is a Neural Network?*
2. **3Blue1Brown** — *Transformers, the tech behind LLMs*
3. **CampusX** — *What is Deep Learning? (Hindi)*
4. **CodeWithHarry** — *All About ML & Deep Learning (Hindi)*

---

## 🏗️ System Architecture

The application is structured into 5 distinct phases:

1. **Chunking (`chunker.py`)**: 
   - Transcripts are parsed into **60-second sliding windows** with a **15-second overlap** to preserve context.
   - Filler words and duplicate speech artifacts are stripped.
2. **Embedding (`embedder.py`)**: 
   - Chunks are embedded using `paraphrase-multilingual-MiniLM-L12-v2`, allowing for cross-lingual vector space mappings (Hindi + English).
   - A **FAISS IndexFlatIP** (Inner Product) is built natively in memory with L2-normalized vectors to replicate Cosine Similarity.
3. **Retrieval (`retriever.py`)**: 
   - Embeds user queries instantly and retrieves the **Top 3** most semantically similar chunks.
   - Attaches a **Confidence Score** (High / Medium / Low).
4. **Generation (`generator.py`)**: 
   - Constructs a strict constraint prompt injecting the retrieved transcripts.
   - Calls the **Cerebras LLM API** to synthesize the final answer.
5. **Interface (`app.py`)**: 
   - A modern **Gradio v6** UI exposing the system. Displays generative answers alongside explicit source citations and timestamps.

---

## 🚀 Quick Setup (Local Development)

### 1. Prerequisites
You will need **Python 3.12+** and [uv](https://github.com/astral-sh/uv) (the blazing-fast Python package installer).

### 2. Environment Variables
You need an API key from Cerebras Cloud to run the Generator phase.
Set the environment variable in your terminal:

**Windows PowerShell:**
```powershell
$env:CEREBRAS_API_KEY="your-api-key-here"
```

**macOS/Linux:**
```bash
export CEREBRAS_API_KEY="your-api-key-here"
```

### 3. Run the App
With `uv` installed, simply run the Gradio app directly. `uv` will automatically manage the dependencies dynamically:

```bash
uv run app.py
```
The server will bind to `localhost:7860`. Open your browser to interact with the UI.

---

## 🤝 Key Design Decisions & Tradeoffs

| Component | Choice | Tradeoff / Rationale |
|-----------|--------|----------------------|
| **Embedding Model** | Multilingual MiniLM | Slightly lower precision than enormous 1B+ parameter models, but effortlessly handles mixed Hindi and English without translation pipelines. |
| **Vector DB** | In-Memory FAISS | No disk persistence between sessions, but enables zero-friction setup and massive speed for few documents. |
| **LLM Inference** | Cerebras `llama3.1-8b` | Instant inference speeds via Cerebras Cloud SDK. |
| **Chunking** | 60s windows / 15s overlap | Optimal balance. Larger chunks dilute precision, smaller chunks lose semantic context. |

---

## 📝 Demo Checklist
*Test these queries to verify source retrieval:*
- *"How does 3Blue1Brown explain the relationship between pixel brightness and a neuron's activation?"* -> Expects `3B1B Neural Network (~0:00 - 5:00)`
- *"What example does CodeWithHarry use to explain supervised learning?"* -> Expects `CodeWithHarry (~5:00)`
- *"How does CampusX explain the difference between ML and Deep Learning?"* -> Expects `CampusX (~2:00)`
