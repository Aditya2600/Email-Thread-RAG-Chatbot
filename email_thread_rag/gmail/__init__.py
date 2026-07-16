"""Stage-3 Gmail connectivity: OAuth, Pub/Sub push, durable delta sync.

Nothing here is imported by the memory RAG path. Every Google/psycopg import
inside this package is function-local, so ``RAG_BACKEND=memory`` never needs
Gmail packages, credentials, Docker, or network access.
"""
