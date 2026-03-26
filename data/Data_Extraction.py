"""
Transcript Fetcher
------------------
Fetches transcripts for four YouTube videos using youtube-transcript-api v1.0+.
The API is instance-based: YouTubeTranscriptApi() must be instantiated first.

Output: transcripts/<video_name>.json
  Each file contains: video_id, name, language, and the segment list
  (each segment: {text, start, duration}).
"""

import json
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

# ── Video list ─────────────────────────────────────────────────────────────────
VIDEOS = [
    {
        "id": "aircAruvnKk",
        "name": "3Blue1Brown_Neural_Network",
        "lang": ["en"],
    },
    {
        "id": "wjZofJX0v4M",
        "name": "3Blue1Brown_Transformers",
        "lang": ["en"],
    },
    {
        "id": "fHF22Wxuyw4",
        "name": "CampusX_Deep_Learning_Hindi",
        "lang": ["en"],   # prefer Hindi, fall back to English
    },
    {
        "id": "C6YtPJxNULA",
        "name": "CodeWithHarry_ML_DL_Hindi",
        "lang": ["en"],
    },
]

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = "transcripts"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # v1.0+: must instantiate the class
    ytt = YouTubeTranscriptApi()

    for video in VIDEOS:
        vid   = video["id"]
        name  = video["name"]
        langs = video["lang"]
        out   = os.path.join(OUT_DIR, f"{name}.json")

        print(f"\n[...] Fetching: {name}  ({vid})")
        try:
            # list() returns a TranscriptList; find_transcript picks best match
            transcript_list = ytt.list(vid)
            try:
                transcript_obj = transcript_list.find_transcript(langs)
            except NoTranscriptFound:
                # Fall back to any available transcript
                transcript_obj = next(iter(transcript_list))

            fetched = transcript_obj.fetch()

            # Convert snippet objects to plain dicts
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched
            ]

            lang_label = (
                f"{transcript_obj.language_code} "
                f"({'generated' if transcript_obj.is_generated else 'manual'})"
            )

            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "video_id": vid,
                        "name": name,
                        "language": lang_label,
                        "transcript": segments,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            total_words = sum(len(s["text"].split()) for s in segments)
            print(
                f"  [OK]  Saved '{out}'  |  "
                f"{len(segments)} segments  |  ~{total_words} words  |  lang: {lang_label}"
            )

        except TranscriptsDisabled:
            print(f"  [FAIL]  Transcripts disabled for {vid}")
        except StopIteration:
            print(f"  [FAIL]  No transcripts available at all for {vid}")
        except Exception as exc:
            print(f"  [FAIL]  {vid}: {exc}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()