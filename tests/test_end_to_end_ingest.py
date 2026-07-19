from __future__ import annotations

import json
from pathlib import Path

from email_thread_rag.rag.engine import RAGEngine
from email_thread_rag.scripts.build_dataset_slice import build_dataset_slice
from email_thread_rag.rag.corpus import ingest_corpus
from email_thread_rag.rag.retrieval import HybridRetriever, load_chunks
from email_thread_rag.rag.reranker import CrossEncoderReranker, OverlapRerankScorer
from email_thread_rag.rag.vector_index import HashingEncoder, VectorIndex


def test_end_to_end_manifest_build_and_ingest(tmp_path, sample_records):
    settings, _, _, _ = sample_records
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "enron"
    mailbox_source = tmp_path / "source-mailbox.json"
    mailbox_source.write_text(
        json.dumps(
            {
                "mailbox": "allen-p",
                "messages": [
                    {
                        "doc_id": "fixture-msg-1",
                        "message_id": "<fixture-msg-1@example.com>",
                        "thread_id": "thread-local",
                        "date": "2024-01-09T09:30:00Z",
                        "from": "alice@enron.com",
                        "to": ["bob@enron.com"],
                        "cc": [],
                        "subject": "Budget Thread",
                        "body": "Please see the attached budget.",
                        "attachments": [
                            {
                                "attachment_id": "fixture-msg-1-att-1",
                                "filename": "fixture_attachment.txt",
                                "local_source": str(fixture_root / "thread-local" / "fixture_attachment.txt"),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "source": {
            "name": "enronarchive/enron-mail",
            "provider": "huggingface_dataset",
            "repo_id": "enronarchive/mail",
            "repo_type": "dataset",
            "revision": "fixture-offline",
        },
        "mailboxes": [
            {
                "mailbox": "allen-p",
                "files": [
                    {
                        "relative_path": "_mailboxes/allen-p/mailbox.json",
                        "local_source": str(mailbox_source),
                    }
                ],
                "selection": {
                    "max_threads": 1,
                    "max_messages_per_thread": 1,
                },
            }
        ],
    }
    settings.dataset_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    from email_thread_rag.scripts import build_dataset_slice as build_module

    build_module.get_settings = lambda: settings
    build_dataset_slice(force=True)
    ingest_corpus(settings)
    loaded_chunks = load_chunks(settings.chunk_store_path)
    retriever = HybridRetriever(
        loaded_chunks,
        settings,
        vector_index=VectorIndex.build(loaded_chunks, settings, encoder=HashingEncoder()),
        reranker=CrossEncoderReranker(settings, scorer=OverlapRerankScorer()),
    )
    engine = RAGEngine(settings, retriever=retriever)
    session = engine.session_store.start_session("allen-p:thread-local")
    outcome = engine.ask(session.session_id, "What amount is in the attachment?", search_outside_thread=False)
    assert "[msg: <fixture-msg-1@example.com>, page: 1]" in outcome.response.answer
