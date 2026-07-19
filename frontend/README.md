# Frontend

The web UI for Inbox Copilot. It talks to the FastAPI backend in
`email_thread_rag/` — there is no mock data.

## Run it locally

Start the backend first (from the repo root):

```bash
uvicorn email_thread_rag.app.main:app --port 8000
```

Then:

```bash
cp .env.example .env   # points at http://localhost:8000
npm install
npm run dev            # http://localhost:8080
```

The app calls the API on relative paths (`/ask`, `/health`, …). In development
the Vite dev server proxies those paths to `VITE_BACKEND_ORIGIN`, so the browser
sees a single origin and the backend needs no CORS policy. The proxied paths are
listed in `vite.config.ts`; add to that list if the backend gains a route.

## Deploying

Relative paths mean the app and the API must answer on the **same origin** in
production — put both behind one reverse proxy, with the API routes forwarded to
the backend. The alternative is a CORS policy on the backend that allows the
frontend's origin; the backend does not ship one today.

`VITE_*` values are compiled into the browser bundle and are public. Only the
backend origin belongs there. OAuth client secrets, provider API keys, database
URLs, and LLM keys stay on the server.

## What the backend does and doesn't provide

Wired to real routes:

| UI | Endpoint |
| --- | --- |
| Sources page | `GET /threads` |
| Ask Inbox | `POST /start_session`, `POST /ask` |
| Gmail availability | `GET /openapi.json` (checks whether the Gmail routes are mounted) |
| Connect Gmail | `GET /gmail/oauth/start` (only shown once this reports available) |

Not available, and left out rather than filled with samples or a "not configured" notice:

- **Sync Activity** — the backend has no sync-history route.
- **Message and attachment browsing** — nothing lists them outside an answer's
  citations.
- **Disconnect Gmail, sync toggles, answer-generation settings** — no routes.
- **API/Gmail status indicators** — no status pills in the UI; a failed ask
  surfaces as a toast instead.
