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


def mount_gmail_routes(app: FastAPI, settings) -> bool:
    """Attach the Stage-3 Gmail routes, but only if Gmail is configured.

    Every import here is function-local on purpose: RAG_BACKEND=memory must
    import this module without the gmail/postgres extras installed. Returns
    whether the routes were mounted.
    """
    if not (settings.gmail_pubsub_subscription and settings.database_url):
        return False

    from email_thread_rag.gmail.cipher import build_token_cipher
    from email_thread_rag.gmail.client import GooglePubSubPushVerifier, HttpxGmailClient
    from email_thread_rag.gmail.oauth import refresh_access_token
    from email_thread_rag.gmail.repository import PostgresSyncStore
    from email_thread_rag.gmail.routes import build_oauth_router
    from email_thread_rag.gmail.webhook import build_router
    from email_thread_rag.rag.paradedb.repository import connect

    # autocommit: no transaction is left open between requests.
    conn = connect(settings.database_url, autocommit=True)
    store_factory = lambda: PostgresSyncStore(conn)  # noqa: E731

    app.include_router(
        build_router(
            store_factory=store_factory,
            verifier=GooglePubSubPushVerifier(
                audience=settings.gmail_pubsub_audience or settings.api_base_url,
                expected_subscription=settings.gmail_pubsub_subscription,
                expected_service_account=settings.gmail_pubsub_service_account,
            ),
        )
    )
    app.include_router(
        build_oauth_router(
            settings=settings,
            store_factory=store_factory,
            client_factory=lambda refresh_token: HttpxGmailClient(
                refresh_access_token(
                    refresh_token=refresh_token,
                    client_id=settings.gmail_client_id,
                    client_secret=settings.gmail_client_secret,
                )
            ),
            cipher=build_token_cipher(settings),
        )
    )

    from fastapi import APIRouter

    sync_router = APIRouter(prefix="/gmail")

    @sync_router.get("/sync-history")
    def sync_history(limit: int = 50) -> dict:
        # gmail_sync_jobs is the queue; each row is one Pub/Sub-driven sync.
        return {"events": store_factory().recent_jobs(limit=max(1, min(limit, 200)))}

    app.include_router(sync_router)
    return True


def start_inline_gmail_worker(app: FastAPI, settings) -> bool:
    """Drain the Gmail sync queue from a background thread in this process.

    Without this, the webhook records jobs into gmail_sync_jobs and nothing ever
    claims them, so email_messages -- and /threads -- stay empty. A build failure
    (missing extras, DB unreachable) is logged and swallowed: the API still
    serves, and a dedicated worker process can pick the jobs up instead.
    """
    import logging

    log = logging.getLogger(__name__)
    if not (settings.gmail_inline_worker and settings.gmail_pubsub_subscription and settings.database_url):
        log.warning(
            "gmail inline worker NOT started (inline_worker=%s pubsub_subscription=%s database_url=%s); "
            "jobs will queue in gmail_sync_jobs but nothing will claim them in-process",
            settings.gmail_inline_worker,
            bool(settings.gmail_pubsub_subscription),
            bool(settings.database_url),
        )
        return False

    from email_thread_rag.gmail.worker import start_inline_worker

    try:
        worker = start_inline_worker(settings)
    except Exception:  # noqa: BLE001 - never let worker startup take down the API
        log.exception("gmail inline worker failed to start")
        return False
    log.info(
        "gmail inline worker started (tenant_id=%s mailbox_id=%s poll=%.1fs)",
        settings.tenant_id,
        settings.mailbox_id,
        settings.gmail_worker_poll_interval,
    )
    app.state.gmail_worker = worker
    app.router.add_event_handler("shutdown", worker.stop)
    return True


mount_gmail_routes(app, settings)
start_inline_gmail_worker(app, settings)


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

