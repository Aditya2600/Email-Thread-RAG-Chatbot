"""Stage 4: asynchronous LLM contextualization of persisted chunks.

Nothing in this package is imported by the memory RAG path, and nothing here is
imported at all unless contextualization is explicitly enabled: psycopg and the
HTTP provider are function-local imports, so `RAG_BACKEND=memory` never pulls in
a database driver or an LLM client. `tests/test_context_independence.py` proves
it rather than trusting it.

The invariant the whole package exists to protect: `text`, `source_start`, and
`source_end` are immutable. Contextualization only ever rewrites `embed_text`
and the embedding derived from it.
"""
