import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Q&A Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag.answer_question(question)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/api/ingest")
def trigger_ingest():
    """Re-scans /data and rebuilds the vector store. Call this after adding new files."""
    count = rag.ingest()
    return {"chunks_ingested": count}


# Serve the frontend (static files) at the root, after API routes are registered
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

