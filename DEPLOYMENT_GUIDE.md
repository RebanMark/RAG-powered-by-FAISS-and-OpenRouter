# 🚀 Deployment Guide

This guide will walk you through how to successfully host your RAG Demo project on **GitHub** (for code sharing and version control) and **Hugging Face Spaces** (for a live, interactive web demo).

---

## 📂 1. Pushing to GitHub (Code Only)

You want to share your project code but avoid uploading local environments (like `.venv`) or heavy data unnecessarily if it isn't required. We've set up a `.gitignore` file to ensure `.venv` is ignored.

### Steps:
1. **Initialize Git**  
   Open your terminal in the `rag-demo` directory and run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: RAG Demo"
   ```
   *(Note: The `.gitignore` file we created automatically prevents `.venv` from being added.)*

2. **Create a Repository on GitHub**  
   - Go to [GitHub.com](https://github.com/new).
   - Create a new repository named `rag-demo` (leave it public or private, do *not* initialize with a README since you already have one).

3. **Push the Code**  
   Copy the commands GitHub provides under "push an existing repository from the command line", which look like this:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/rag-demo.git
   git push -u origin main
   ```

🎉 Your code is now live on GitHub!

---

## ⚙️ 2. Deploying to Hugging Face Spaces (Live App)

Hugging Face Spaces provides extremely easy, free hosting for Gradio Python apps.

### Steps:
1. **Create a New Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - **Space name:** `rag-demo` (or similar).
   - **License:** MIT (or whatever you prefer).
   - **Select the Space SDK:** Choose **Gradio**.
   - **Hardware:** Blank/Free (CPU basic).
   - Click **Create Space**.

2. **Upload Your Files**
   You can either push via Git or simply upload files manually using the browser UI:
   - Click **Files and versions** tab in your new Space.
   - Click **Add file** -> **Upload files**.
   - Select and drag inside all the contents of your `rag-demo` directory:
     - `app.py`
     - `chunker.py`
     - `embedder.py`
     - `generator.py`
     - `retriever.py`
     - `requirements.txt`
     - `README.md`
     - And the `data/` folder containing your 4 JSON transcripts.
     *(Note: Do **NOT** upload `.venv`)*
   - Commit changes to the `main` branch.

3. **Add Your API Key (Crucial!)**
   For your RAG app to generate answers, you must securely provide your Cerebras API key so it's not public in the code.
   - Go to your Space's **Settings** tab.
   - Scroll down to **Variables and secrets**.
   - Click **New secret**.
   - **Name:** `CEREBRAS_API_KEY`
   - **Value:** *[paste your actual Cerebras API key here]*
   - Click **Save**.

4. **Watch it Build!**
   - Navigate back to the **App** tab.
   - Hugging Face will read your `requirements.txt`, install everything automatically, and launch your Gradio interface.
   - Once the status changes from "Building" to "Running", you have a fully public RAG web app!

---
*Happy deploying!*
