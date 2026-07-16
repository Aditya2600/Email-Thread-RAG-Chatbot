from __future__ import annotations

from datetime import datetime, timezone

from email_thread_rag.app.schemas import ChunkRecord, EmailRecord
from email_thread_rag.rag.chunking import EmailAwareChunker, chunk_email
from email_thread_rag.rag.email_segmentation import build_embed_text, segment_email_body


def _email(body: str, **overrides) -> EmailRecord:
    base = dict(
        doc_id="msg-1",
        message_id="<msg-1@acme.com>",
        thread_id="thread_123",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="Bob <bob@acme.com>",
        to=["sarah@acme.com"],
        cc=["finance@acme.com"],
        subject="Q3 Budget",
        body_text=body,
        source_path="/tmp/msg-1.eml",
        source_type="fixture",
        in_reply_to="<parent@acme.com>",
    )
    base.update(overrides)
    return EmailRecord(**base)


RAW = (
    "Hi Sarah,\n\n"
    "The approved budget is now $120,000. Please proceed with the vendor.\n\n"
    "Thanks,\n"
    "Bob\n"
    "--\n"
    "Bob Smith | Finance Dept | bob@acme.com\n\n"
    "This e-mail is confidential and intended only for the addressee.\n\n"
    "On Tue, May 2, Sarah wrote:\n"
    "> What is the final approved number?\n"
    "> Thanks\n"
)


def test_short_authored_email_is_one_chunk():
    chunks = chunk_email(_email(RAW))
    assert len(chunks) == 1
    assert "$120,000" in chunks[0].text


def test_quoted_history_excluded():
    chunk = chunk_email(_email(RAW))[0]
    assert "final approved number" not in chunk.text
    assert "final approved number" not in (chunk.embed_text or "")


def test_signature_and_disclaimer_excluded():
    chunk = chunk_email(_email(RAW))[0]
    assert "Bob Smith" not in chunk.text
    assert "confidential" not in chunk.text.lower()


def test_text_is_clean_authored_evidence():
    chunk = chunk_email(_email(RAW))[0]
    # No injected headers in text.
    assert not chunk.text.startswith("From:")
    assert "Thread-ID:" not in chunk.text
    assert "Subject:" not in chunk.text


def test_embed_text_contains_metadata_headers():
    chunk = chunk_email(_email(RAW))[0]
    embed = chunk.embed_text
    assert "From: Bob <bob@acme.com>" in embed
    assert "To: sarah@acme.com" in embed
    assert "Cc: finance@acme.com" in embed
    assert "Date: 2026-05-03" in embed
    assert "Subject: Q3 Budget" in embed
    assert "Thread-ID: thread_123" in embed
    assert embed.rstrip().endswith(chunk.text)


def test_cc_not_populated_from_to():
    # cc empty, to present -> Cc line must be absent, To line present.
    chunk = chunk_email(_email(RAW, cc=[]))[0]
    assert "To: sarah@acme.com" in chunk.embed_text
    assert "Cc:" not in chunk.embed_text


def test_source_offsets_map_to_authored_text():
    email = _email(RAW)
    chunk = chunk_email(email)[0]
    authored = segment_email_body(RAW).authored_text
    assert chunk.source_start is not None and chunk.source_end is not None
    assert authored[chunk.source_start : chunk.source_end] == chunk.text


def test_long_email_splits_and_retains_all_authored_content():
    paras = [f"Point number {i}: " + "budget detail word " * 40 for i in range(12)]
    body = "\n\n".join(paras)
    email = _email(body)
    chunks = chunk_email(email)
    assert len(chunks) > 1
    joined = " ".join(c.text for c in chunks)
    for i in range(12):
        assert f"Point number {i}:" in joined
    # Each chunk stays within a sane token budget (allowing overlap slack).
    assert all(c.token_count <= EmailAwareChunker().max_tokens + 60 for c in chunks)


def test_embed_text_defaults_to_text_for_legacy_records():
    # Old fixture/record without embed_text: validator backfills it to text.
    chunk = ChunkRecord(
        chunk_id="legacy-0",
        doc_id="d",
        thread_id="t",
        message_id="m",
        kind="email",
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        text="legacy body",
        token_count=2,
        source_path="/tmp/x",
        source_type="fixture",
    )
    assert chunk.embed_text == "legacy body"
    # Round-trips through JSON (as load_chunks does) preserving the default.
    reloaded = ChunkRecord.model_validate_json(chunk.model_dump_json())
    assert reloaded.embed_text == "legacy body"


def test_build_embed_text_omits_absent_fields():
    out = build_embed_text("body only")
    assert out == "body only"


def test_quote_only_email_produces_no_chunks():
    # A pure forward/reply with no new authored words must not leak the
    # quoted history into the normal retrieval path as if it were authored.
    body = "On Tue, May 2, Sarah wrote:\n> What is the final approved number?\n> Thanks\n"
    email = _email(body)
    chunks = chunk_email(email)
    assert chunks == []
    assert email.quoted_text and "final approved number" in email.quoted_text
