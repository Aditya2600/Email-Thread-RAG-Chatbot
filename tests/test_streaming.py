from __future__ import annotations


def test_streaming_response_shape(api_client):
    start = api_client.post("/start_session", json={"thread_id": "thread-alpha"})
    session_id = start.json()["session_id"]
    response = api_client.post(
        "/ask",
        json={"session_id": session_id, "text": "What amount is in budget_final.pdf?", "search_outside_thread": False},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: delta" in body
    assert "event: final" in body
    assert '"trace_id"' in body

