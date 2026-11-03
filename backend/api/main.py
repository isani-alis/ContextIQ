from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import AsyncIterator, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from backend.agent.graph import (
    AgentState,
    build_graph,
    build_retrieval_graph,
    stream_answer,
)
from backend.ingestion.parser import ingest_file
from backend.retrieval.vector_store import VectorStore

load_dotenv()

app = FastAPI(title="ContextIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_store: VectorStore | None = None
_graph = None
_retrieval_graph = None
_llm: ChatOllama | None = None

# Session memory: session_id → message history
_sessions: Dict[str, List[dict]] = {}


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="llama3.2", temperature=0)
    return _llm


def get_graph():
    global _graph, _store
    if _graph is None:
        if VectorStore.exists():
            _store = VectorStore.load()
        else:
            raise HTTPException(status_code=400, detail="No documents ingested yet. Upload files first.")
        _graph = build_graph(_store, get_llm())
    return _graph


def get_retrieval_graph():
    global _retrieval_graph, _store
    if _retrieval_graph is None:
        if VectorStore.exists():
            _store = VectorStore.load()
        else:
            raise HTTPException(status_code=400, detail="No documents ingested yet. Upload files first.")
        _retrieval_graph = build_retrieval_graph(_store, get_llm())
    return _retrieval_graph


# ------------------------------------------------------------------
# /ingest
# ------------------------------------------------------------------

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    global _store, _graph, _retrieval_graph

    all_chunks = []
    saved_files = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".csv", ".xlsx", ".xls"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(dest)

        chunks = ingest_file(dest)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in uploaded files.")

    for idx, chunk in enumerate(all_chunks):
        chunk.chunk_id = idx

    _store = VectorStore()
    _store.build(all_chunks)
    _graph = None
    _retrieval_graph = None

    return {
        "status": "ok",
        "files": [f.name for f in saved_files],
        "chunks_indexed": len(all_chunks),
    }


# ------------------------------------------------------------------
# /query — blocking (kept for RAGAS eval)
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.post("/query")
async def query(req: QueryRequest):
    graph = get_graph()
    history = _sessions.get(req.session_id, [])

    initial_state: AgentState = {
        "query": req.question,
        "condensed_query": req.question,
        "history": history,
        "candidates": [],
        "reranked": [],
        "confident": False,
        "answer": "",
        "citations": [],
        "context": "",
    }

    result = graph.invoke(initial_state)

    history = history + [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": result["answer"]},
    ]
    _sessions[req.session_id] = history[-20:]

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "confident": result["confident"],
        "condensed_query": result["condensed_query"],
        "session_id": req.session_id,
    }


# ------------------------------------------------------------------
# /query/stream — SSE streaming endpoint
# Protocol:
#   event: citations  → JSON payload with citations + confident flag
#   event: token      → one LLM token at a time
#   event: done       → stream finished
#   event: error      → something went wrong
# ------------------------------------------------------------------

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    retrieval_graph = get_retrieval_graph()
    llm = get_llm()
    history = _sessions.get(req.session_id, [])

    initial_state: AgentState = {
        "query": req.question,
        "condensed_query": req.question,
        "history": history,
        "candidates": [],
        "reranked": [],
        "confident": False,
        "answer": "",
        "citations": [],
        "context": "",
    }

    async def event_generator() -> AsyncIterator[str]:
        # Step 1: run retrieval pipeline (condense → retrieve → rerank → prepare_context)
        try:
            state = retrieval_graph.invoke(initial_state)
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        # Step 2: send citations immediately so frontend can render them
        yield (
            f"event: citations\n"
            f"data: {json.dumps({'citations': state['citations'], 'confident': state['confident'], 'condensed_query': state['condensed_query']})}\n\n"
        )

        # Step 3: stream LLM tokens
        full_answer = ""
        try:
            async for token in stream_answer(state, llm):
                full_answer += token
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        yield "event: done\ndata: {}\n\n"

        # Step 4: update session memory after stream completes
        updated_history = history + [
            {"role": "user", "content": req.question},
            {"role": "assistant", "content": full_answer},
        ]
        _sessions[req.session_id] = updated_history[-20:]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# /session/{session_id}
# ------------------------------------------------------------------

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    return {"history": _sessions.get(session_id, []), "session_id": session_id}


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": VectorStore.exists(),
        "chunks": len(_store.chunks) if _store else 0,
    }
