"""
src/ingest.py
─────────────────────────────────────────────────────────────────
Responsibility: Load the IT knowledge base JSON, convert each
issue-solution pair into a LangChain Document, embed them using
a HuggingFace sentence-transformer, and persist them into a
local Chroma vector store.

Run this ONCE (or whenever your JSON data changes) before
starting the chatbot.
─────────────────────────────────────────────────────────────────
"""

import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Paths ────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # project root
DATA_PATH   = BASE_DIR / "data" / "it_issues.json"
VECTORDB_DIR = str(BASE_DIR / "vectordb")             # Chroma persists here

# ── Embedding model (runs locally, no API key needed) ────────────
#    'all-MiniLM-L6-v2' is fast, lightweight, and great for semantic
#    similarity on short technical paragraphs.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(json_path: Path) -> list[Document]:
    """
    Read it_issues.json and convert each entry into a LangChain
    Document object.

    Document structure:
      - page_content : the text the retriever will search over
                       (we combine issue + symptoms + solution so
                        the vector captures the full context)
      - metadata     : structured fields for filtering / display
    """
    with open(json_path, "r", encoding="utf-8") as f:
        issues = json.load(f)

    documents = []

    for idx, item in enumerate(issues):
        # ── Build rich page_content ───────────────────────────────
        # Joining issue, symptoms, and solution into one block gives
        # the embedding model maximum semantic context. This means a
        # user saying "my laptop won't boot" will still match the
        # "computer won't turn on" entry because the surrounding
        # words reinforce the meaning.
        symptoms_str = ", ".join(item.get("symptoms", []))

        page_content = (
            f"Issue: {item['issue']}\n"
            f"Category: {item['category']}\n"
            f"Symptoms: {symptoms_str}\n"
            f"Solution:\n{item['solution']}"
        )

        # ── Attach metadata ───────────────────────────────────────
        # Metadata is NOT embedded — it's stored alongside the vector
        # so you can filter by category later (e.g., show only Network
        # issues) or surface it in the UI.
        metadata = {
            "doc_id"  : idx,
            "issue"   : item["issue"],
            "category": item["category"],
            "symptoms": symptoms_str,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"[Ingest] Loaded {len(documents)} documents from {json_path}")
    return documents
def get_splitter_for_book(filename: str) -> RecursiveCharacterTextSplitter:
    """
    Different books need different chunking strategies.

    Dummies All-in-One:
      - Very long chapters (20-40 pages each)
      - Has sub-sections with headers like "Understanding Your RAM"
      - Contains numbered steps, tips, warnings, tables
      - Needs LARGER chunks to preserve step sequences
      - Needs HIGH overlap so steps don't get cut across chunks

    IT Troubleshooting Manual:
      - Shorter, more concise sections
      - Already structured in Q&A / problem-solution format
      - Smaller chunks work better — each section is self-contained

    Small Business IT Guide:
      - Very short sections
      - Minimal overlap needed
    """

    filename_lower = filename.lower()

    # ── Dummies All-in-One ────────────────────────────────────────
    if "dummies" in filename_lower or "all-in-one" in filename_lower or "aio" in filename_lower:
        print(f"  → Detected: Dummies All-in-One — using large chunk strategy")
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,        # larger — preserves full step sequences
            chunk_overlap=200,      # high overlap — avoids cutting mid-procedure
            separators=[
                "\n\n\n",           # major section breaks
                "\n\n",             # paragraph breaks
                "\n",               # line breaks
                ". ",               # sentence boundaries
                " ",                # word boundaries
            ],
        )

    # ── IT Troubleshooting Manual ─────────────────────────────────
    elif "troubleshooting manual" in filename_lower or "learning made easy" in filename_lower:
        print(f"  → Detected: IT Troubleshooting Manual — using medium chunk strategy")
        return RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " "],
        )

    # ── Small Business Guide ──────────────────────────────────────
    elif "small business" in filename_lower or "smb" in filename_lower:
        print(f"  → Detected: Small Business Guide — using small chunk strategy")
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " "],
        )

    # ── Default fallback for unknown PDFs ─────────────────────────
    else:
        print(f"  → Unknown book type — using balanced default strategy")
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "],
        )


def clean_chunk_text(text: str) -> str:
    """
    PDFs often extract with messy artifacts:
      - Multiple spaces from column layouts
      - Hyphen line-breaks from justified text  (trou-\nbleshooting)
      - Page headers/footers repeated every page
      - Stray single characters from formatting
      - Tab characters

    This cleans the raw extracted text before embedding so the
    vectors are cleaner and retrieval is more accurate.
    """
    import re

    # Rejoin hyphenated line breaks (e.g. "trou-\nbleshooting")
    text = re.sub(r'-\n(\w)', r'\1', text)

    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove lines that are just page numbers or very short (headers/footers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip: empty, page numbers, single words that are likely headers
        if len(stripped) < 4:
            continue
        # Skip: lines that are ALL CAPS and short (likely chapter headers
        # repeated as running headers — keep if long enough to be content)
        if stripped.isupper() and len(stripped) < 30:
            continue
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def load_pdf_manuals(manuals_dir: Path) -> list[Document]:
    """
    Load all PDF manuals with per-book chunking strategy and
    text cleaning pipeline.
    """
    if not manuals_dir.exists():
        print(f"[Ingest] No manuals/ directory found — skipping PDF ingestion.")
        return []

    pdf_files = list(manuals_dir.glob("*.pdf"))
    if not pdf_files:
        print("[Ingest] No PDF files found in manuals/ — skipping.")
        return []

    print(f"[Ingest] Found {len(pdf_files)} PDF manual(s) — loading...\n")

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"[Ingest] ── Processing: {pdf_path.name}")

        try:
            # ── Load raw pages ────────────────────────────────────
            loader = PyPDFLoader(str(pdf_path))
            pages  = loader.load()
            print(f"  → {len(pages)} pages loaded")

            # ── Clean each page's text ────────────────────────────
            for page in pages:
                page.page_content = clean_chunk_text(page.page_content)

            # Remove pages that became empty after cleaning
            pages = [p for p in pages if len(p.page_content.strip()) > 50]
            print(f"  → {len(pages)} pages after cleaning")

            # ── Get the right splitter for this book ──────────────
            splitter = get_splitter_for_book(pdf_path.name)
            chunks   = splitter.split_documents(pages)

            # ── Filter out low-quality chunks ─────────────────────
            # Chunks under 80 chars are usually stray headers or
            # page artifacts — not useful for retrieval
            chunks = [c for c in chunks if len(c.page_content.strip()) > 80]

            # ── Enrich chunk metadata ─────────────────────────────
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source"    : pdf_path.name,
                    "category"  : "IT Manual",
                    "type"      : "manual",
                    "chunk_id"  : i,
                    "page"      : chunk.metadata.get("page", "?"),
                })

            all_chunks.extend(chunks)

            print(f"  → {len(chunks)} clean chunks ready for embedding\n")

        except Exception as e:
            print(f"  ⚠ Failed to load {pdf_path.name}: {e}\n")

    print(f"[Ingest] Total PDF chunks across all manuals: {len(all_chunks)}")
    return all_chunks

def build_vectorstore_combined(
    json_documents: list[Document],
    pdf_chunks: list[Document],
) -> Chroma:
    """
    Embed and persist BOTH JSON issues and PDF manual chunks
    into the same Chroma collection.

    Keeping them in one collection means a single retriever
    query searches both sources simultaneously — the best
    match wins regardless of source type.
    """
    all_documents = json_documents + pdf_chunks

    print(f"[Ingest] Total documents to embed: {len(all_documents)}")
    print(f"         ├── JSON issues : {len(json_documents)}")
    print(f"         └── PDF chunks  : {len(pdf_chunks)}")

    print(f"[Ingest] Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[Ingest] Embedding all documents — this may take a moment...")
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=VECTORDB_DIR,
        collection_name="it_support_kb",
    )

    print(f"[Ingest] ✅ Combined vector store persisted → {VECTORDB_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load an ALREADY-PERSISTED Chroma collection from disk.

    Call this at chatbot startup (after ingest has run at least once)
    so you skip the expensive embedding step every time.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings,
        collection_name="it_support_kb",
    )

    count = vectorstore._collection.count()
    print(f"[Ingest] Loaded existing vector store ({count} vectors) from {VECTORDB_DIR}")
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 3):
    """
    Wrap the Chroma store as a LangChain retriever.

    k=3  →  return the 3 most semantically similar documents.
            Raising k gives more context but increases LLM token cost.
    search_type="similarity" uses cosine distance on the embeddings.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run this whenever you add new JSON entries OR new PDF manuals:
        python -m src.ingest

    It will REBUILD the entire vector store from scratch so
    everything stays in sync.
    """
    MANUALS_DIR = BASE_DIR / "data" / "manuals"

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {DATA_PATH}")

    # ── Load both sources ─────────────────────────────────────────
    json_docs  = load_documents(DATA_PATH)
    pdf_chunks = load_pdf_manuals(MANUALS_DIR)

    # ── Wipe old vectordb and rebuild clean ───────────────────────
    # Important: always rebuild from scratch when adding new sources
    # to avoid stale or duplicate vectors
    import shutil
    if Path(VECTORDB_DIR).exists():
        print("[Ingest] Removing old vector store — rebuilding clean...")
        shutil.rmtree(VECTORDB_DIR)

    # ── Build combined store ──────────────────────────────────────
    vs = build_vectorstore_combined(json_docs, pdf_chunks)

    # ── Sanity check ──────────────────────────────────────────────
    print("\n[Ingest] Sanity check — querying: 'Lenovo battery not charging'")
    results = vs.similarity_search("Lenovo battery not charging", k=3)
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "JSON KB")
        cat    = r.metadata.get("category", "General")
        print(f"\n  Result {i}: [{cat}] from '{source}'")
        print(f"  Preview : {r.page_content[:120]}...")

    print(f"\n✅ Ingestion complete.")
    print(f"   JSON issues : {len(json_docs)}")
    print(f"   PDF chunks  : {len(pdf_chunks)}")
    print(f"   Total vectors: {len(json_docs) + len(pdf_chunks)}")