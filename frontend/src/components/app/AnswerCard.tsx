import { motion } from "framer-motion";
import type { Answer, Citation } from "@/lib/api";
import { ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CitationCard } from "./CitationCard";

const statusMeta: Record<Answer["status"], { label: string; className: string } | null> = {
  grounded: { label: "Grounded answer", className: "bg-success/12 text-success" },
  deterministic: null,
  abstained: { label: "Not enough evidence", className: "bg-warning/12 text-warning" },
};

export function AnswerCard({
  answer,
  onOpenCitation,
}: {
  answer: Answer;
  onOpenCitation: (c: Citation) => void;
}) {
  const hasCitations = answer.citations.length > 0;
  const status = statusMeta[answer.status];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-4"
    >
      <div className="glass-card rounded-2xl p-6">
        {status && (
          <div className="mb-3">
            <Badge className={`gap-1.5 rounded-full border-0 ${status.className}`}>
              <ShieldCheck className="h-3 w-3" />
              {status.label}
            </Badge>
          </div>
        )}

        <p className="whitespace-pre-wrap text-[17px] leading-relaxed text-ink">
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
