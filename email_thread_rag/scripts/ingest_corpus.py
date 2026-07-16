from __future__ import annotations

import argparse
import json
from pathlib import Path

from email_thread_rag.config import get_settings
from email_thread_rag.rag.bm25_index import BM25Index
from email_thread_rag.rag.corpus import ingest_corpus
from email_thread_rag.rag.utils import write_json
from email_thread_rag.rag.vector_index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize emails/attachments and build indexes.")
    parser.add_argument("--build-slice", action="store_true", help="Force rebuilding the raw dataset slice before ingestion.")
    args = parser.parse_args()
    settings = get_settings()

    if args.build_slice or not settings.resolved_manifest_path.exists():
        from email_thread_rag.scripts.build_dataset_slice import build_dataset_slice

        build_dataset_slice(force=args.build_slice)

    emails, _, chunks, stats = ingest_corpus(settings)
    bm25_index = BM25Index(chunks)
    bm25_index.save(settings.index_dir / "bm25.pkl")
    vector_index = VectorIndex.build(chunks, settings)
    vector_index.save(settings.index_dir / "vector.pkl")
    write_json(settings.index_dir / "index_stats.json", stats)

    if settings.rag_backend == "paradedb":
        from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
        from email_thread_rag.rag.paradedb.repository import connect, verify_extensions

        conn = connect(settings.database_url)
        try:
            verify_extensions(conn)
            paradedb_stats = persist_corpus_to_paradedb(
                conn,
                emails,
                chunks,
                tenant_id=settings.tenant_id,
                mailbox_id=settings.mailbox_id,
                encoder=vector_index.encoder,
                embedding_dim=settings.embedding_dim,
            )
            conn.commit()
        finally:
            conn.close()
        stats["paradedb"] = paradedb_stats

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
