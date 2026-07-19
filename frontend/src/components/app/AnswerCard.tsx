import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { Answer, Citation } from "@/lib/api";
import {
  ShieldCheck,
  ChevronDown,
  Sparkles,
  Search,
  GitBranch,
  ListOrdered,
  Compass,
  ScrollText,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CitationCard } from "./CitationCard";

// Labels come from the backend's reported retrieval routes; anything new falls
// back to a neutral icon rather than being dropped.
const methodMeta: Record<string, { icon: typeof Search; tint: string }> = {
  "Keyword match": { icon: Search, tint: "text-brand" },
  "Semantic match": { icon: Sparkles, tint: "text-brand-2" },
  "Graph evidence": { icon: GitBranch, tint: "text-brand-2" },
  Reranked: { icon: ListOrdered, tint: "text-success" },
  "Searched outside thread": { icon: Compass, tint: "text-warning" },
};

const statusMeta: Record<
  Answer["status"],
  { label: string; className: string; icon: typeof ShieldCheck }
> = {
  grounded: {
    label: "Grounded answer",
    className: "bg-success/12 text-success",
    icon: ShieldCheck,
  },
  deterministic: {
    label: "Evidence-only answer",
    className: "bg-brand/12 text-brand",
    icon: ScrollText,
  },
  abstained: {
    label: "Not enough evidence",
    className: "bg-warning/12 text-warning",
    icon: ShieldCheck,
  },
};

export function AnswerCard({
  answer,
  onOpenCitation,
}: {
  answer: Answer;
  onOpenCitation: (c: Citation) => void;
}) {
  const [howOpen, setHowOpen] = useState(false);
  const hasCitations = answer.citations.length > 0;
  const status = statusMeta[answer.status];
  const StatusIcon = status.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-4"
    >
      <div className="glass-card rounded-2xl p-6">
        <div className="flex flex-wrap items-center gap-2">
          <Badge className={`gap-1.5 rounded-full border-0 ${status.className}`}>
            <StatusIcon className="h-3 w-3" />
            {status.label}
          </Badge>
          {hasCitations && (
            <span className="text-xs text-ink-muted">
              Based on {answer.citations.length} source{answer.citations.length === 1 ? "" : "s"}
            </span>
          )}
          {answer.status === "deterministic" && (
            <span className="text-xs text-ink-muted">
              · answer generation is off, so this is assembled from cited evidence
            </span>
          )}
        </div>

        <p className="mt-3 whitespace-pre-wrap text-[17px] leading-relaxed text-ink">
          {answer.answer}
          {hasCitations && (
            <span className="ml-1 inline-flex items-center gap-0.5 align-baseline">
              {answer.citations.map((c) => (
                <button
                  key={c.id}
                  onClick={() => onOpenCitation(c)}
                  className="ml-0.5 inline-grid h-5 w-5 place-items-center rounded-md bg-accent text-[11px] font-semibold text-brand hover:bg-brand hover:text-white transition-colors"
                >
                  {c.index}
                </button>
              ))}
            </span>
          )}
        </p>

        {answer.methods.length > 0 && (
          <div className="mt-4">
            <button
              onClick={() => setHowOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-muted hover:text-ink transition-colors"
            >
              <Sparkles className="h-3.5 w-3.5" />
              How it was found
              <ChevronDown
                className={`h-3.5 w-3.5 transition-transform ${howOpen ? "rotate-180" : ""}`}
              />
            </button>
            <AnimatePresence initial={false}>
              {howOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="mt-2 flex flex-wrap gap-2">
                    {answer.methods.map((m) => {
                      const meta = methodMeta[m] ?? { icon: Sparkles, tint: "text-brand" };
                      const Icon = meta.icon;
                      return (
                        <div
                          key={m}
                          className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-ink"
                        >
                          <Icon className={`h-3 w-3 ${meta.tint}`} />
                          {m}
                        </div>
                      );
                    })}
                  </div>
                  {answer.rewrite && (
                    <p className="mt-2 text-xs text-ink-muted">
                      Searched for &ldquo;{answer.rewrite}&rdquo;
                      {answer.rewriteMode ? ` · ${answer.rewriteMode} rewrite` : ""}
                    </p>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {answer.status === "abstained" && (
          <p className="mt-5 text-xs text-ink-muted">
            Narrow the question to a specific person, project, amount, or date range and ask again.
          </p>
        )}
      </div>

      {hasCitations && (
        <div className="grid gap-3 lg:hidden">
          {answer.citations.map((c) => (
            <CitationCard key={c.id} citation={c} onOpen={onOpenCitation} />
          ))}
        </div>
      )}
    </motion.div>
  );
}
