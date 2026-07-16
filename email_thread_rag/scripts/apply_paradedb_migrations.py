from __future__ import annotations

import argparse

from email_thread_rag.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ParadeDB schema migrations (idempotent).")
    parser.add_argument("--database-url", default=None, help="Overrides DATABASE_URL/settings.database_url.")
    args = parser.parse_args()

    from email_thread_rag.rag.paradedb.repository import apply_migrations, connect, verify_extensions

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    conn = connect(database_url)
    try:
        versions = verify_extensions(conn)
        print(f"Extensions verified: {versions}")
        applied = apply_migrations(conn)
        if applied:
            print(f"Applied migrations: {applied}")
        else:
            print("No pending migrations.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
