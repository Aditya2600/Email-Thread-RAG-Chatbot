import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowUp } from "lucide-react";
import type { Citation } from "@/lib/api";
import { useAsk } from "@/lib/queries";
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

function AskInboxPage() {
  const [question, setQuestion] = useState("");
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const ask = useAsk();
  const reduceMotion = useReducedMotion();
  const answer = ask.data ?? null;

  const idle = !answer && !ask.isPending;
  const hasSources = answer !== null && answer.citations.length > 0;

  // The composer is centred on its own, so making room for the panel has to move
  // it — there is no position that satisfies both. Holding the width across a
  // follow-up keeps that move to once per session instead of once per question.
  const [heldOpen, setHeldOpen] = useState(false);
  useEffect(() => {
    if (!ask.isPending) setHeldOpen(hasSources);
  }, [ask.isPending, hasSources]);
  const roomForSources = ask.isPending ? heldOpen : hasSources;

  // Nothing on the right until there is something real to put there. The one
  // exception is a follow-up, where the column is already open and a skeleton
  // reads better than a hole.
  const showPanel = hasSources || (ask.isPending && heldOpen);

  const submit = (q: string) => {
    if (!q.trim() || ask.isPending) return;
    setQuestion(q);
    ask.mutate(q);
  };

  const openCitation = (c: Citation) => {
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

  const listVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: reduceMotion ? 0 : 0.05,
        delayChildren: reduceMotion ? 0 : 0.08,
      },
    },
  };

  const cardVariants = reduceMotion
    ? { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { duration: 0.15 } } }
    : {
        hidden: { opacity: 0, x: 12 },
        visible: { opacity: 1, x: 0, transition: { duration: 0.24, ease: "easeOut" as const } },
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
        {/* Fixed width so widening the page slides this column without reflowing it. */}
        <div className="w-full min-w-0 lg:w-[720px] lg:shrink-0">
          {idle && (
            <div className="mb-7">
              <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-ink">
                Ask Inbox
              </h1>
              <p className="mt-1.5 text-sm text-ink-muted">Search your email and files.</p>
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
                disabled={!question.trim() || ask.isPending}
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

          <div className="mt-6">
            {ask.isPending && <AnswerSkeleton />}
            {answer && !ask.isPending && (
              <AnswerCard answer={answer} onOpenCitation={openCitation} />
            )}
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
                  {hasSources && (
                    <span className="text-[11px] text-ink-muted">{answer.citations.length}</span>
                  )}
                </div>

                {/* No `initial={false}`: the list is present on this boundary's
                    first render, and suppressing it would kill the stagger on
                    the very first question. */}
                <AnimatePresence mode="wait">
                  {ask.isPending ? (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      <CitationSkeletonList />
                    </motion.div>
                  ) : (
                    <motion.ul
                      key="sources-list"
                      variants={listVariants}
                      initial="hidden"
                      animate="visible"
                      className="space-y-3"
                    >
                      {answer?.citations.map((c) => (
                        <motion.li key={c.id} variants={cardVariants}>
                          <CitationCard citation={c} onOpen={openCitation} />
                        </motion.li>
                      ))}
                    </motion.ul>
                  )}
                </AnimatePresence>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>

      <SourceDrawer citation={activeCitation} open={drawerOpen} onOpenChange={setDrawerOpen} />
    </motion.div>
  );
}
