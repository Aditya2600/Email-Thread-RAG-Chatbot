import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, Sparkles, AlertTriangle, Clock, PlugZap } from "lucide-react";
import { ApiError, type Citation } from "@/lib/api";
import { useAsk } from "@/lib/queries";
import { AnswerCard } from "@/components/app/AnswerCard";
import { CitationCard } from "@/components/app/CitationCard";
import { AnswerSkeleton, CitationSkeletonList } from "@/components/app/Skeletons";
import { SourceDrawer } from "@/components/app/SourceDrawer";

export const Route = createFileRoute("/")({
  component: AskInboxPage,
  head: () => ({
    meta: [
      { title: "Ask Inbox · Inbox Copilot" },
      {
        name: "description",
        content:
          "Ask anything across your email and attachments. Every answer comes back with exact, cited evidence.",
      },
    ],
  }),
});

// Prompt starters, not sample results — nothing here is presented as an answer.
const exampleQuestions = [
  "What was the last decision in this thread?",
  "What amount was approved, and who approved it?",
  "What did the attachment say about the deadline?",
  "What changed since the earlier proposal?",
];

function AskInboxPage() {
  const [question, setQuestion] = useState("");
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const ask = useAsk();
  const answer = ask.data ?? null;

  const submit = (q: string) => {
    if (!q.trim() || ask.isPending) return;
    setQuestion(q);
    ask.mutate(q);
  };

  const openCitation = (c: Citation) => {
    setActiveCitation(c);
    setDrawerOpen(true);
  };

  const idle = !answer && !ask.isPending && !ask.isError;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      className="mx-auto flex w-full max-w-[1400px] gap-6 px-4 md:px-8 py-8"
    >
      {/* Main column */}
      <div className="min-w-0 flex-1">
        {idle && (
          <div className="mb-6">
            <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-xs text-ink-muted">
              <Sparkles className="h-3 w-3 text-brand" />
              Grounded in your inbox
            </div>
            <h1 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight text-ink">
              Ask <span className="text-gradient-brand">Inbox</span>
            </h1>
            <p className="mt-2 max-w-xl text-[15px] text-ink-muted">
              Ask anything across your email and attachments. Every answer comes back with exact,
              cited evidence — never a hallucinated summary.
            </p>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(question);
          }}
          className="glass-card rounded-2xl p-2 shadow-[var(--shadow-soft)] focus-within:ring-2 focus-within:ring-ring/40 transition"
        >
          <div className="flex items-start gap-2 px-3 pt-3">
            <Sparkles className="mt-1 h-4 w-4 shrink-0 text-brand" />
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(question);
                }
              }}
              rows={2}
              placeholder="Ask anything across your email and attachments…"
              className="min-h-[52px] w-full resize-none bg-transparent text-[15px] text-ink placeholder:text-ink-muted focus:outline-none"
            />
          </div>
          <div className="flex items-center justify-end px-2 pb-2 pt-1">
            <button
              type="submit"
              disabled={!question.trim() || ask.isPending}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-brand text-white shadow-[var(--shadow-glow)] transition-all hover:brightness-110 disabled:opacity-40 disabled:shadow-none"
            >
              <ArrowUp className="h-4 w-4" strokeWidth={2.5} />
            </button>
          </div>
        </form>

        {idle && (
          <div className="mt-5">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
              Try an example
            </div>
            <div className="flex flex-wrap gap-2">
              {exampleQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => submit(q)}
                  className="rounded-full border border-border bg-surface-raised px-3.5 py-1.5 text-sm text-ink hover:bg-accent hover:border-brand/30 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="mt-6">
          {ask.isPending && <AnswerSkeleton />}
          {ask.isError && <AskError error={ask.error} onRetry={() => submit(question)} />}
          {answer && !ask.isPending && <AnswerCard answer={answer} onOpenCitation={openCitation} />}
        </div>
      </div>

      {/* Evidence panel */}
      <aside className="hidden lg:block w-[380px] shrink-0">
        <div className="sticky top-20">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-ink">Evidence</h2>
            {answer && answer.citations.length > 0 && (
              <span className="text-[11px] text-ink-muted">{answer.citations.length} sources</span>
            )}
          </div>
          {idle && (
            <div className="glass-card rounded-xl p-6 text-center">
              <div className="mx-auto grid h-10 w-10 place-items-center rounded-lg bg-accent">
                <Sparkles className="h-4 w-4 text-brand" />
              </div>
              <p className="mt-3 text-sm text-ink">Citations appear here</p>
              <p className="mt-1 text-xs text-ink-muted">
                Ask a question to see grounded evidence from your inbox.
              </p>
            </div>
          )}
          {ask.isPending && <CitationSkeletonList />}
          {answer && !ask.isPending && answer.citations.length > 0 && (
            <div className="space-y-3">
              {answer.citations.map((c) => (
                <CitationCard key={c.id} citation={c} onOpen={openCitation} />
              ))}
            </div>
          )}
          {answer && !ask.isPending && answer.citations.length === 0 && (
            <div className="glass-card rounded-xl p-6 text-center">
              <p className="text-sm text-ink">No supported evidence</p>
              <p className="mt-1 text-xs text-ink-muted">
                Nothing in the indexed mail matched closely enough to cite.
              </p>
            </div>
          )}
          {ask.isError && (
            <div className="glass-card rounded-xl p-6 text-center">
              <p className="text-sm text-ink">No evidence to show</p>
              <p className="mt-1 text-xs text-ink-muted">
                The request didn't complete, so nothing was retrieved.
              </p>
            </div>
          )}
        </div>
      </aside>

      <SourceDrawer citation={activeCitation} open={drawerOpen} onOpenChange={setDrawerOpen} />
    </motion.div>
  );
}

/** Failure is reported exactly as the API reported it — never softened into an
 *  answer, and never accompanied by invented citations. */
function AskError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const api = error instanceof ApiError ? error : null;
  const providerDisabled = api?.status === 503;

  const {
    icon: Icon,
    title,
    body,
  } = providerDisabled
    ? {
        icon: PlugZap,
        title: "Answering is unavailable",
        body: api?.message ?? "The backend reported that the answer provider is not available.",
      }
    : api?.kind === "timeout"
      ? {
          icon: Clock,
          title: "The answer took too long",
          body: "The backend didn't respond in time. Ask again, or narrow the question.",
        }
      : api?.kind === "offline"
        ? {
            icon: PlugZap,
            title: "Can't reach the API",
            body: "Inbox Copilot's backend isn't responding. Check that it's running, then try again.",
          }
        : {
            icon: AlertTriangle,
            title: "The request failed",
            body: api?.message ?? "Something went wrong on the way to the backend.",
          };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="rounded-2xl border border-destructive/25 bg-destructive/[0.03] p-6"
    >
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <p className="mt-1 text-sm text-ink-muted">{body}</p>
          {api?.status !== undefined && api.kind === "http" && (
            <p className="mt-1 text-xs text-ink-muted">HTTP {api.status}</p>
          )}
          <button
            onClick={onRetry}
            className="mt-3 inline-flex items-center justify-center rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-medium text-ink hover:bg-accent transition-colors"
          >
            Try again
          </button>
        </div>
      </div>
    </motion.div>
  );
}
