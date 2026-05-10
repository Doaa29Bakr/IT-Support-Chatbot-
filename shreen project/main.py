"""
main.py
─────────────────────────────────────────────────────────────────
Entry point. Run this to launch the full chatbot.

Usage:
    python main.py

What it does:
  1. Checks the vector store exists (reminds you to run ingest first)
  2. Launches Gradio on http://localhost:7860
  3. share=False  → local only (set True to get a public ngrok URL)
─────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Pre-flight checks ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
VECTORDB    = BASE_DIR / "vectordb"
GROQ_KEY    = os.getenv("GROQ_API_KEY")

if not VECTORDB.exists():
    raise RuntimeError(
        "\n❌ Vector store not found!\n"
        "   Run ingestion first:  python -m src.ingest\n"
    )

if not GROQ_KEY:
    raise RuntimeError(
        "\n❌ GROQ_API_KEY not set!\n"
        "   Add it to your .env file:  GROQ_API_KEY=your_key_here\n"
        "   Get a free key at:  https://console.groq.com\n"
    )

# ── Launch ────────────────────────────────────────────────────────
from src.ui import build_ui

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Alex — IT Support Chatbot")
    print("  http://localhost:7860")
    print("="*55 + "\n")

    from src.ui import build_ui, CSS          # ← import CSS too
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,                              # ← moved here in Gradio 6
    )