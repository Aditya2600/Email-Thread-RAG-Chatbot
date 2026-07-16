"""Stage-2 ParadeDB/Postgres backend.

Only imported when ``Settings.rag_backend == "paradedb"``. Nothing in the
memory backend or its tests touches this package, so `psycopg` need not be
installed for `pip install -e '.[dev]'` or the default test run.
"""
