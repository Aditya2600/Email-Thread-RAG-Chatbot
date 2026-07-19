import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, FileText, Mail, ScanLine, ExternalLink } from "lucide-react";
import type { Citation } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { Badge } from "@/components/ui/badge";

export function CitationCard({
  citation,
  onOpen,
}: {
  citation: Citation;
  onOpen: (c: Citation) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const isAttachment = citation.type === "attachment";

  // Entrance is driven by whatever renders the list, so the cards can stagger.
  return (
    <motion.article layout className="glass-card rounded-xl p-4 lift-on-hover">
      <div className="flex items-start gap-3">
        <div className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-brand text-[11px] font-semibold text-white">
          {citation.index}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-ink-muted">
            <span className="inline-flex items-center gap-1">
              {isAttachment ? <FileText className="h-3 w-3" /> : <Mail className="h-3 w-3" />}
              <span className="font-medium text-ink">
                {citation.sender ?? (isAttachment ? "Attachment" : "Email message")}
              </span>
            </span>
            {citation.date && (
              <>
                <span>·</span>
                <span>{formatDate(citation.date)}</span>
              </>
            )}
          </div>
          {citation.subject && (
            <h3 className="mt-0.5 truncate text-sm font-medium text-ink">{citation.subject}</h3>
          )}

          {citation.attachment && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Badge
                variant="secondary"
                className="gap-1 rounded-md bg-accent text-accent-foreground border-0"
              >
                <FileText className="h-3 w-3" />
                {citation.attachment.filename
                  ? `${citation.attachment.filename} · Page ${citation.attachment.page}`
                  : `Page ${citation.attachment.page}`}
              </Badge>
              {citation.attachment.ocr && (
                <Badge
                  variant="outline"
                  className="gap-1 rounded-md border-warning/40 text-warning"
                >
                  <ScanLine className="h-3 w-3" />
                  OCR-derived
                </Badge>
              )}
            </div>
          )}

          <blockquote
            className={`mt-2.5 rounded-lg border-l-2 border-brand/50 bg-surface px-3 py-2 text-sm text-ink ${isAttachment ? "font-mono text-[13px]" : "italic"}`}
          >
            &ldquo;{citation.quote}&rdquo;
          </blockquote>

          <div className="mt-2.5 flex items-center justify-between">
            {citation.context ? (
              <button
                onClick={() => setExpanded((v) => !v)}
                className="inline-flex items-center gap-1 text-xs text-ink-muted hover:text-ink transition-colors"
              >
                <ChevronDown
                  className={`h-3.5 w-3.5 transition-transform ${expanded ? "rotate-180" : ""}`}
                />
                {expanded ? "Hide context" : "Show context"}
              </button>
            ) : (
              <span />
            )}
            <button
              onClick={() => onOpen(citation)}
              className="inline-flex items-center gap-1 text-xs font-medium text-brand hover:opacity-80 transition-opacity"
            >
              Open source <ExternalLink className="h-3 w-3" />
            </button>
          </div>

          <AnimatePresence initial={false}>
            {expanded && citation.context && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <p className="mt-2 rounded-lg bg-muted px-3 py-2 text-xs leading-relaxed text-ink-muted">
                  {citation.context}
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.article>
  );
}
