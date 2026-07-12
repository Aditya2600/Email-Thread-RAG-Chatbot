from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from email_thread_rag.app.schemas import AskRequest, ResetSessionRequest, StartSessionRequest, SwitchThreadRequest
from email_thread_rag.app.streaming import format_sse, iter_text_deltas
from email_thread_rag.config import get_settings
from email_thread_rag.rag.engine import RAGEngine


settings = get_settings()
engine = RAGEngine(settings)
app = FastAPI(title="Email Thread RAG Chatbot")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/threads")
def list_threads() -> dict:
    return {"threads": engine.available_threads()}


@app.post("/start_session")
def start_session(payload: StartSessionRequest) -> dict:
    if engine.available_threads() and payload.thread_id not in engine.available_threads():
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {payload.thread_id}")
    session = engine.session_store.start_session(payload.thread_id)
    return {"session_id": session.session_id, "thread_id": session.thread_id}


@app.post("/switch_thread")
def switch_thread(payload: SwitchThreadRequest) -> dict:
    session = engine.session_store.switch_thread(payload.session_id, payload.thread_id)
    return {"session_id": session.session_id, "thread_id": session.thread_id}


@app.post("/reset_session")
def reset_session(payload: ResetSessionRequest) -> dict:
    session = engine.session_store.reset(payload.session_id)
    return {"session_id": session.session_id, "thread_id": session.thread_id}


@app.post("/ask")
def ask(payload: AskRequest, request: Request):
    accept_header = request.headers.get("accept", "")
    outcome = engine.ask(
        payload.session_id,
        payload.text,
        search_outside_thread=payload.search_outside_thread,
    )

    if "text/event-stream" in accept_header:
        response_payload = outcome.response.model_dump(mode="json")

        def event_stream():
            for delta in iter_text_deltas(outcome.response.answer, settings.answer_stream_chunk_size):
                yield format_sse("delta", {"text": delta, "trace_id": outcome.response.trace_id})
            yield format_sse("final", response_payload)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return JSONResponse(outcome.response.model_dump(mode="json"))

