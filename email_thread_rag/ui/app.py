from __future__ import annotations

import json
from typing import Any

import gradio as gr
import httpx

from email_thread_rag.config import get_settings


settings = get_settings()


def load_threads() -> tuple[gr.Dropdown, str]:
    try:
        response = httpx.get(f"{settings.api_base_url}/threads", timeout=10.0)
        response.raise_for_status()
        threads = response.json().get("threads", [])
        return gr.update(choices=threads, value=threads[0] if threads else None), json.dumps(
            {"threads": threads},
            indent=2,
        )
    except Exception as exc:
        return gr.update(choices=[], value=None), json.dumps({"error": str(exc)}, indent=2)


def clear_thread_context(thread_id: str) -> tuple[str, str, list[dict[str, str]], str]:
    if not thread_id:
        return "", "", [], json.dumps({"status": "Select a thread, then click Start session."}, indent=2)
    return (
        "",
        "",
        [],
        json.dumps(
            {
                "status": "Thread changed. Start a new session before asking.",
                "thread_id": thread_id,
            },
            indent=2,
        ),
    )


def start_session(thread_id: str) -> tuple[str, str, list[dict[str, str]], str]:
    if not thread_id:
        message = json.dumps({"error": "Select a thread before starting a session."}, indent=2)
        return "", "", [], message
    try:
        response = httpx.post(f"{settings.api_base_url}/start_session", json={"thread_id": thread_id}, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        return payload["session_id"], payload["session_id"], [], json.dumps(payload, indent=2)
    except Exception as exc:
        return "", "", [], json.dumps({"error": f"Failed to start session: {exc}"}, indent=2)


def stream_chat(message: str, history: list[dict[str, str]] | None, session_id: str, search_outside_thread: bool):
    history = history or []
    if not session_id:
        yield history, json.dumps({"error": "Start a session by selecting a thread first."}, indent=2)
        return

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    final_payload: dict[str, Any] = {}
    try:
        with httpx.stream(
            "POST",
            f"{settings.api_base_url}/ask",
            json={
                "session_id": session_id,
                "text": message,
                "search_outside_thread": search_outside_thread,
            },
            headers={"Accept": "text/event-stream"},
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            current_event = None
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line.replace("event: ", "", 1).strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.replace("data: ", "", 1))
                if current_event == "delta":
                    updated_history[-1]["content"] += payload.get("text", "")
                    yield updated_history, json.dumps({"status": "streaming...", "chars_received": len(updated_history[-1]["content"])}, indent=2)
                elif current_event == "final":
                    final_payload = payload
    except httpx.HTTPError as exc:
        yield updated_history, json.dumps({"error": f"SSE request failed: {exc}"}, indent=2)
        return

    debug = {
        "rewrite": final_payload.get("rewrite"),
        "rewrite_mode": final_payload.get("rewrite_mode"),
        "retrieved": [
            {
                "chunk_id": item["chunk"]["chunk_id"],
                "score": item["metrics"]["chunk_support_score"],
                "rerank_score_norm": item["metrics"]["rerank_score_norm"],
            }
            for item in final_payload.get("retrieved", [])
        ],
        "citations": final_payload.get("citations", []),
        "trace_id": final_payload.get("trace_id"),
        "outside_thread_used": final_payload.get("outside_thread_used"),
        "metrics": final_payload.get("metrics", {}),
    }
    yield updated_history, json.dumps(debug, indent=2)


with gr.Blocks(title="Email Thread RAG") as demo:
    gr.Markdown("# Email Thread RAG")
    with gr.Row():
        thread_selector = gr.Dropdown(label="Thread selector", choices=[], allow_custom_value=True)
        refresh_button = gr.Button("Refresh threads")
        start_button = gr.Button("Start session")
    session_state = gr.State("")
    session_display = gr.Textbox(label="Session ID", interactive=False, visible=False)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat")
            user_message = gr.Textbox(label="Ask", placeholder="Type your question and press Enter…")
            search_toggle = gr.Checkbox(label="Search outside thread", value=False)
        debug_panel = gr.Code(label="Debug", language="json")

    refresh_button.click(load_threads, outputs=[thread_selector, debug_panel])
    thread_selector.change(
        clear_thread_context,
        inputs=[thread_selector],
        outputs=[session_state, session_display, chatbot, debug_panel],
    )
    start_button.click(
        start_session,
        inputs=[thread_selector],
        outputs=[session_state, session_display, chatbot, debug_panel],
    )
    user_message.submit(
        stream_chat,
        inputs=[user_message, chatbot, session_state, search_toggle],
        outputs=[chatbot, debug_panel],
    ).then(lambda: "", outputs=[user_message])
    demo.load(load_threads, outputs=[thread_selector, debug_panel])


if __name__ == "__main__":
    demo.launch(server_name=settings.ui_host, server_port=settings.ui_port)
