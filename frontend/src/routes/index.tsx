import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowUp, Plus } from "lucide-react";
import type { Answer, Citation } from "@/lib/api";
import { askInbox, newConversation } from "@/lib/api";
import { AnswerCard } from "@/components/app/AnswerCard";
import { CitationCard } from "@/components/app/CitationCard";
import { AnswerSkeleton, CitationSkeletonList } from "@/components/app/Skeletons";
import { SourceDrawer } from "@/components/app/SourceDrawer";

export const Route = createFileRoute("/")({
  component: AskInboxPage,
  head: () => ({
    meta: [{ title: "Ask Inbox · Inbox Copilot" }],
  }),
});

// Prompt starters, not sample results — nothing here is presented as an answer.
const exampleQuestions = [
  "What was the last decision in this thread?",
  "What amount was approved, and who approved it?",
  "What did the attachment say about the deadline?",
];

// One exchange in the conversation. Each turn owns its own answer and citations;
// nothing is ever merged across turns.
type ChatTurn = {
  id: string;
  question: string;
  answer: Answer | null;
  error: string | null;
  pending: boolean;
};

function AskInboxPage() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  // Which turn's sources fill the right panel. Defaults to the latest turn.
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const reduceMotion = useReducedMotion();

  const busy = turns.some((t) => t.pending);
  const idle = turns.length === 0;

  const selected = turns.find((t) => t.id === selectedId) ?? null;
  const selectedHasSources = (selected?.answer?.citations.length ?? 0) > 0;
  // Panel is open when the selected turn has sources, or while it is still
  // running (a skeleton reads better than the column snapping in on arrival).
  const showPanel = selectedHasSources || (selected?.pending ?? false);
  const roomForSources = showPanel;

  const submit = (q: string) => {
    if (!q.trim() || busy) return; // one in-flight turn at a time
    const id =
      typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    setTurns((prev) => [...prev, { id, question: q, answer: null, error: null, pending: true }]);
    setSelectedId(id); // sources default to the newest turn
    setQuestion("");

    askInbox(q)
      .then((answer) =>
        setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, answer, pending: false } : t))),
      )
      .catch(() =>
        setTurns((prev) =>
          prev.map((t) =>
            t.id === id ? { ...t, error: "Couldn’t get an answer. Try again.", pending: false } : t,
          ),
        ),
      );
  };

  const startNewChat = () => {
    newConversation(); // drops the backend session, clearing its turn history
    setTurns([]);
    setSelectedId(null);
    setQuestion("");
    setActiveCitation(null);
    setDrawerOpen(false);
  };

  // Selecting a turn is what routes its citations into the panel; opening the
  // drawer shows the one clicked. The two stay in lock-step so the panel never
  // mixes citations from different turns.
  const openCitation = (turnId: string, c: Citation) => {
    setSelectedId(turnId);
    setActiveCitation(c);
    setDrawerOpen(true);
  };

  const panelMotion = reduceMotion
    ? {
        initial: { opacity: 0 },
        animate: { opacity: 1, transition: { duration: 0.15 } },
        exit: { opacity: 0, transition: { duration: 0.1 } },
      }
    : {
        initial: { opacity: 0, x: 16 },
        animate: { opacity: 1, x: 0, transition: { duration: 0.26, ease: "easeOut" as const } },
        exit: { opacity: 0, x: 16, transition: { duration: 0.18, ease: "easeIn" as const } },
      };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      className={`mx-auto w-full max-w-[784px] px-4 md:px-8 py-10 transition-[max-width] duration-300 ease-out ${
        roomForSources ? "lg:max-w-[1188px]" : "lg:max-w-[784px]"
      }`}
    >
      <div className="flex gap-6">
        <div className="w-full min-w-0 lg:w-[720px] lg:shrink-0">
          {idle ? (
            <div className="mb-7">
              <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-ink">
                Ask Inbox
              </h1>
              <p className="mt-1.5 text-sm text-ink-muted">Search your email and files.</p>
            </div>
          ) : (
            <div className="mb-5 flex items-center justify-between">
              <h1 className="text-lg font-semibold tracking-tight text-ink">Ask Inbox</h1>
              <button
                onClick={startNewChat}
                className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm text-ink-muted hover:text-ink hover:bg-surface transition-colors"
              >
                <Plus className="h-3.5 w-3.5" strokeWidth={2.5} />
                New chat
              </button>
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit(question);
            }}
            className="glass-card rounded-2xl p-2 focus-within:ring-2 focus-within:ring-ring/40 transition"
          >
            <div className="flex items-start gap-2 px-3 pt-3">
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
                placeholder="Ask a question…"
                className="min-h-[52px] w-full resize-none bg-transparent text-[15px] text-ink placeholder:text-ink-muted focus:outline-none"
              />
            </div>
            <div className="flex items-center justify-end px-2 pb-2 pt-1">
              <button
                type="submit"
                disabled={!question.trim() || busy}
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-brand text-white transition-opacity hover:opacity-90 disabled:opacity-40"
              >
                <ArrowUp className="h-4 w-4" strokeWidth={2.5} />
              </button>
            </div>
          </form>

          {idle && (
            <div className="mt-5 flex flex-wrap gap-2">
              {exampleQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => submit(q)}
                  className="rounded-full px-3.5 py-1.5 text-sm text-ink-muted hover:text-ink hover:bg-surface transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* All turns, oldest first. Earlier turns stay visible as new ones
              arrive; loading/error live on their own turn only. */}
          <div className="mt-6 space-y-8">
            {turns.map((turn) => (
              <div key={turn.id} className="space-y-3">
                <div className="flex justify-end">
                  <p className="max-w-[90%] rounded-2xl bg-brand px-4 py-2 text-[15px] text-white">
                    {turn.question}
                  </p>
                </div>

                {turn.pending && <AnswerSkeleton />}

                {turn.error && (
                  <div className="glass-card rounded-2xl p-5">
                    <p className="text-sm text-warning">{turn.error}</p>
                    <button
                      onClick={() => {
                        setTurns((prev) => prev.filter((t) => t.id !== turn.id));
                        submit(turn.question);
                      }}
                      disabled={busy}
                      className="mt-3 text-sm font-medium text-brand hover:opacity-80 disabled:opacity-40"
                    >
                      Retry
                    </button>
                  </div>
                )}

                {turn.answer && (
                  <div
                    onClickCapture={() => setSelectedId(turn.id)}
                    className={`rounded-2xl transition ${
                      turn.id === selectedId && selectedHasSources ? "ring-2 ring-ring/30" : ""
                    }`}
                  >
                    <AnswerCard
                      answer={turn.answer}
                      onOpenCitation={(c) => openCitation(turn.id, c)}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <AnimatePresence initial={false}>
          {showPanel && (
            <motion.aside
              key="sources"
              {...panelMotion}
              className="hidden lg:block w-[380px] shrink-0"
            >
              <div className="sticky top-10">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-ink">Sources</h2>
                  {selectedHasSources && (
                    <span className="text-[11px] text-ink-muted">
                      {selected?.answer?.citations.length}
                    </span>
                  )}
                </div>

                {selected?.pending ? (
                  <CitationSkeletonList />
                ) : (
                  <ul className="space-y-3">
                    {selected?.answer?.citations.map((c) => (
                      <li key={c.id}>
                        <CitationCard
                          citation={c}
                          onOpen={(cit) => openCitation(selected.id, cit)}
                        />
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>

      <SourceDrawer citation={activeCitation} open={drawerOpen} onOpenChange={setDrawerOpen} />
    </motion.div>
  );
}
