// The single place where Inbox Copilot's backend contract is translated into UI
// models. Nothing here invents data: a field the backend did not send stays
// `undefined` so the UI can hide it instead of showing a placeholder.
//
// Discovered backend surface (FastAPI, see email_thread_rag/app/main.py):
//   GET  /health
//   GET  /threads
//   POST /start_session   { thread_id }
//   POST /switch_thread   { session_id, thread_id }
//   POST /reset_session   { session_id }
//   POST /ask             { session_id, text, search_outside_thread }
//   GET  /gmail/oauth/start?tenant_id&mailbox_id   (mounted only when Gmail is configured)
//   GET  /gmail/oauth/callback                     (browser lands here from Google)
//   POST /gmail/pubsub/push                        (server-to-server, not for the UI)
//
// Requests are relative. In development the Vite proxy forwards them to
// VITE_BACKEND_ORIGIN; in production the app and the API must share an origin
// (reverse proxy) or the backend must allow the frontend origin via CORS.

const ASK_TIMEOUT_MS = 45_000;
const DEFAULT_TIMEOUT_MS = 10_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly kind: "http" | "timeout" | "offline",
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new ApiError("The request took too long to come back.", "timeout");
    }
    throw new ApiError("Can't reach the Inbox Copilot API.", "offline");
  }
  if (!res.ok) {
    throw new ApiError(await readDetail(res), "http", res.status);
  }
  return (await res.json()) as T;
}

async function readDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Non-JSON error body; fall through to the status line.
  }
  return `The API returned ${res.status}.`;
}

function postJson<T>(path: string, body: unknown, timeoutMs?: number): Promise<T> {
  return request<T>(
    path,
    { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) },
    timeoutMs,
  );
}

// --- Backend wire types ------------------------------------------------------

type WireChunk = {
  chunk_id: string;
  thread_id: string;
  message_id: string;
  kind: "email" | "attachment";
  attachment_name: string | null;
  page_no: number | null;
  sender: string | null;
  date: string;
  subject: string | null;
  text: string;
  ocr_used: boolean;
};

type WireHit = {
  chunk: WireChunk;
  source_lists: string[];
  rerank_rank: number | null;
};

type WireCitation = {
  message_id: string;
  page_no: number | null;
  chunk_id: string;
  clause_text: string;
  clause_support_score: number;
  formatted: string;
};

type WireAskResponse = {
  answer: string;
  citations: WireCitation[];
  rewrite: string;
  rewrite_mode: string;
  retrieved: WireHit[];
  trace_id: string;
  outside_thread_used: boolean;
  metrics: {
    answer_support_score: number;
    citation_coverage: number;
    evidence_count: number;
    top_thread_support_score: number;
  };
  answer_status: "answered" | "abstained" | null;
};

// --- UI models ---------------------------------------------------------------

export type Citation = {
  id: string;
  index: number;
  type: "email" | "attachment";
  quote: string;
  messageId: string;
  /** Undefined whenever the backend did not return retrieval metadata. */
  sender?: string;
  subject?: string;
  date?: string;
  threadLabel?: string;
  context?: string;
  attachment?: { filename?: string; page: number; ocr?: boolean };
};

/**
 * `deterministic` is the backend's default answering path: no answer-generation
 * provider is enabled, so the answer is assembled from validated evidence
 * rather than generated. It is a mode, not a failure.
 */
export type AnswerStatus = "grounded" | "abstained" | "deterministic";

export type Answer = {
  id: string;
  question: string;
  answer: string;
  status: AnswerStatus;
  /** Retrieval routes the backend actually reported. Empty when it reported none. */
  methods: string[];
  citations: Citation[];
  rewrite?: string;
  rewriteMode?: string;
  outsideThreadUsed: boolean;
  evidenceCount: number;
  traceId: string;
};

// --- Adapters ----------------------------------------------------------------

const METHOD_LABELS: Record<string, string> = {
  bm25: "Keyword match",
  dense: "Semantic match",
  graph: "Graph evidence",
};

/** `[budget.pdf, page: 2 (OCR)]` — the only attachment identity the grounded
 *  path returns, since it sends `retrieved: []`. The deterministic path writes
 *  `[msg: <id>, page: N]` instead, which carries no filename, so it's excluded. */
const FORMATTED_ATTACHMENT = /^\[(?!msg: )(.+), page: (\d+)(\s\(OCR\))?\]$/;

function toCitation(wire: WireCitation, index: number, chunks: Map<string, WireChunk>): Citation {
  const chunk = chunks.get(wire.chunk_id);
  const base: Citation = {
    id: `${wire.chunk_id}:${index}`,
    index: index + 1,
    type: wire.page_no !== null ? "attachment" : "email",
    quote: wire.clause_text,
    messageId: wire.message_id,
  };

  if (chunk) {
    return {
      ...base,
      type: chunk.kind === "attachment" ? "attachment" : "email",
      sender: chunk.sender ?? undefined,
      subject: chunk.subject ?? undefined,
      date: chunk.date,
      threadLabel: chunk.thread_id,
      context: chunk.text,
      attachment:
        chunk.page_no !== null
          ? {
              filename: chunk.attachment_name ?? undefined,
              page: chunk.page_no,
              ocr: chunk.ocr_used,
            }
          : undefined,
    };
  }

  const parsed = FORMATTED_ATTACHMENT.exec(wire.formatted);
  if (parsed) {
    base.attachment = { filename: parsed[1], page: Number(parsed[2]), ocr: Boolean(parsed[3]) };
    base.type = "attachment";
  }
  return base;
}

function toMethods(wire: WireAskResponse): string[] {
  const routes = new Set<string>();
  let reranked = false;
  for (const hit of wire.retrieved) {
    for (const route of hit.source_lists) routes.add(METHOD_LABELS[route] ?? route);
    if (hit.rerank_rank !== null) reranked = true;
  }
  const methods = [...routes];
  if (reranked) methods.push("Reranked");
  if (wire.outside_thread_used) methods.push("Searched outside thread");
  return methods;
}

function toAnswer(wire: WireAskResponse, question: string): Answer {
  const chunks = new Map(wire.retrieved.map((hit) => [hit.chunk.chunk_id, hit.chunk]));
  return {
    id: wire.trace_id,
    question,
    answer: wire.answer,
    status:
      wire.answer_status === "abstained"
        ? "abstained"
        : wire.answer_status === "answered"
          ? "grounded"
          : "deterministic",
    methods: toMethods(wire),
    citations: wire.citations.map((c, i) => toCitation(c, i, chunks)),
    rewrite: wire.rewrite || undefined,
    rewriteMode: wire.rewrite_mode || undefined,
    outsideThreadUsed: wire.outside_thread_used,
    evidenceCount: wire.metrics.evidence_count,
    traceId: wire.trace_id,
  };
}

// --- Session -----------------------------------------------------------------

// /ask needs a session, and a session needs a thread. One session per browser
// tab is enough; it is re-created if the backend forgets it.
let sessionPromise: Promise<string> | null = null;

async function openSession(): Promise<string> {
  const { threads } = await request<{ threads: string[] }>("/threads");
  const { session_id } = await postJson<{ session_id: string; thread_id: string }>(
    "/start_session",
    {
      thread_id: threads[0] ?? "default",
    },
  );
  return session_id;
}

function getSession(): Promise<string> {
  sessionPromise ??= openSession().catch((err) => {
    sessionPromise = null;
    throw err;
  });
  return sessionPromise;
}

// --- Public API --------------------------------------------------------------

/** Thread IDs the backend has indexed. This is the only inventory route it has:
 *  there is no endpoint that lists individual messages or attachments. */
export async function listThreads(): Promise<string[]> {
  const { threads } = await request<{ threads: string[] }>("/threads");
  return threads;
}

function postAsk(sessionId: string, question: string): Promise<WireAskResponse> {
  return postJson<WireAskResponse>(
    "/ask",
    { session_id: sessionId, text: question, search_outside_thread: true },
    ASK_TIMEOUT_MS,
  );
}

export async function askInbox(question: string): Promise<Answer> {
  const sessionId = await getSession();
  try {
    return toAnswer(await postAsk(sessionId, question), question);
  } catch (err) {
    // The backend restarted and dropped in-memory sessions: open a new one once.
    if (err instanceof ApiError && (err.status === 404 || err.status === 422)) {
      sessionPromise = null;
      return toAnswer(await postAsk(await getSession(), question), question);
    }
    throw err;
  }
}

/**
 * Gmail routes are mounted only when the backend has a Pub/Sub subscription and
 * a database configured, so their presence in the schema is the honest test for
 * whether the Connect button can do anything.
 */
export async function getGmailAvailability(): Promise<{ oauthAvailable: boolean }> {
  const schema = await request<{ paths: Record<string, unknown> }>("/openapi.json");
  return { oauthAvailable: "/gmail/oauth/start" in schema.paths };
}

export async function startGmailAuthorization(params: {
  tenantId: string;
  mailboxId: string;
}): Promise<{ authorizationUrl: string }> {
  const query = new URLSearchParams({ tenant_id: params.tenantId, mailbox_id: params.mailboxId });
  const { authorization_url } = await request<{ authorization_url: string }>(
    `/gmail/oauth/start?${query}`,
  );
  // Only ever navigate to a URL the backend produced, and only over HTTPS.
  const url = new URL(authorization_url);
  if (url.protocol !== "https:") {
    throw new ApiError("The backend returned an authorization URL that isn't HTTPS.", "http");
  }
  return { authorizationUrl: url.toString() };
}
