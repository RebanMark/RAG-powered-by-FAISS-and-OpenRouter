"""
chunker.py — Phase 1
Splits transcript JSON into 60-second windows with 15-second overlap.
"""

import re
import json
from pathlib import Path

# Maps filename stem → human-readable video name (used as prefix in embeddings)
VIDEO_NAME_MAP = {
    "3Blue1Brown_Neural_Network":   "3Blue1Brown - But what is a Neural Network?",
    "3Blue1Brown_Transformers":     "3Blue1Brown - Transformers, the tech behind LLMs",
    "CampusX_Deep_Learning_Hindi":  "CampusX - What is Deep Learning? (Hindi)",
    "CodeWithHarry_ML_DL_Hindi":    "CodeWithHarry - All About ML & Deep Learning",
}

FILLER_PATTERN = re.compile(
    r'\b(uh+|um+|you know|like)\b',
    flags=re.IGNORECASE
)

WINDOW_SECONDS  = 60
OVERLAP_SECONDS = 15


def _format_time(seconds: float) -> str:
    """Convert seconds to mm:ss string."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _clean_text(text: str) -> str:
    """Remove filler words, extra whitespace, and simple consecutive duplicates."""
    text = FILLER_PATTERN.sub('', text)
    # Collapse multiple spaces / newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove consecutive duplicate words (e.g. "the the")
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    return text


def chunk_transcript(transcript: list[dict], video_name: str) -> list[dict]:
    """
    Given a list of {text, start, duration} objects and a human-readable
    video_name, return a list of chunk dicts with sliding windows.
    """
    chunks = []
    start_time = 0.0

    while True:
        window_end = start_time + WINDOW_SECONDS
        # Collect all caption lines that fall within [start_time, window_end)
        lines = [
            entry["text"]
            for entry in transcript
            if entry["start"] >= start_time and entry["start"] < window_end
        ]

        if not lines:
            # No more content in this window — check if we've passed the end
            max_start = max(e["start"] for e in transcript)
            if start_time > max_start:
                break
            # Advance and try next window
            start_time += (WINDOW_SECONDS - OVERLAP_SECONDS)
            continue

        raw_text = " ".join(lines)
        cleaned  = _clean_text(raw_text)

        if cleaned:
            # Prefix text with video name for better embedding context
            prefixed_text = f"{video_name}: {cleaned}"

            chunks.append({
                "text":                prefixed_text,
                "video_name":          video_name,
                "start_time":          start_time,
                "start_time_formatted": _format_time(start_time),
            })

        start_time += (WINDOW_SECONDS - OVERLAP_SECONDS)

        # Stop when the window start moves past the last caption
        max_start = max(e["start"] for e in transcript)
        if start_time > max_start:
            break

    return chunks


def load_and_chunk_all(data_dir: str | Path) -> list[dict]:
    """
    Load all 4 transcript JSON files from data_dir and return a flat list
    of all chunks across every video.
    """
    data_dir = Path(data_dir)
    all_chunks = []

    for stem, video_name in VIDEO_NAME_MAP.items():
        filepath = data_dir / f"{stem}.json"
        if not filepath.exists():
            print(f"[WARNING] File not found: {filepath}")
            continue

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        transcript = data.get("transcript", [])
        chunks     = chunk_transcript(transcript, video_name)
        all_chunks.extend(chunks)
        print(f"[chunker] {video_name!r}: {len(chunks)} chunks")

    print(f"[chunker] Total chunks: {len(all_chunks)}")
    return all_chunks


# ── Quick test when run directly ────────────────────────────────────────────
if __name__ == "__main__":
    chunks = load_and_chunk_all(Path(__file__).parent / "data")
    print("\nSample chunk:")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
