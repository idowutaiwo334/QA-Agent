"""
Core RAG (Retrieval-Augmented Generation) engine.

Handles:
- Loading documents from the /data folder (txt, md, pdf)
- Splitting them into overlapping chunks
- Embedding chunks with a local sentence-transformers model (no API key needed)
- Storing/querying vectors in a persistent local Chroma database
- Calling Claude with retrieved context to answer questions
"""

import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from anthropic import Anthropic

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = chromadb.PersistentClient(path=str(DB_DIR))


def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )


# ---------- Loading & chunking ----------

def _read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents() -> list[dict]:
    """Reads every supported file in /data and returns [{source, text}, ...]"""
    docs = []
    if not DATA_DIR.exists():
        return docs

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in (".txt", ".md"):
                text = _read_txt_or_md(path)
            elif suffix == ".pdf":
                text = _read_pdf(path)
            else:
                continue
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        text = text.strip()
        if text:
            docs.append({"source": str(path.relative_to(DATA_DIR)), "text": text})

    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker on whitespace-normalized text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest() -> int:
    """Loads all documents from /data, chunks them, and (re)builds the vector store.
    Returns the number of chunks stored."""
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR}. Add .txt, .md, or .pdf files there.")
        return 0

    # Reset the collection so re-running ingest doesn't duplicate chunks
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = get_collection()

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['source']}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})

    if not texts:
        return 0

    # Batch add (Chroma handles embedding internally via the embedding function)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Ingested {len(docs)} document(s) into {len(texts)} chunks.")
    return len(texts)


# ---------- Retrieval + generation ----------

_anthropic_client = None


def get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy backend/.env.example to backend/.env "
                "and add your key."
            )
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


SYSTEM_PROMPT = """You are a helpful Q&A assistant that answers questions using ONLY the provided context.

Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly \
instead of guessing.
- Be concise and direct.
- When useful, mention which source(s) your answer draws from."""


def answer_question(question: str, top_k: int | None = None) -> dict:
    """Retrieves relevant chunks and asks Claude to answer using them.
    Returns {"answer": str, "sources": [str], "chunks_used": int}"""
    top_k = top_k or int(os.environ.get("TOP_K", 4))
    collection = get_collection()

    if collection.count() == 0:
        return {
            "answer": (
                "No documents have been ingested yet. Add files to the /data folder "
                "and run `python ingest.py`, then ask again."
            ),
            "sources": [],
            "chunks_used": 0,
        }

    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = sorted(set(m["source"] for m in metadatas))

    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}]\n{c}" for c, m in zip(chunks, metadatas)
    )

    client = get_anthropic_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return {"answer": answer_text, "sources": sources, "chunks_used": len(chunks)}
