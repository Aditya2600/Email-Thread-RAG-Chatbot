import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import type { Citation } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { FileText, Mail, ScanLine, User } from "lucide-react";

export function SourceDrawer({
  citation,
  open,
  onOpenChange,
}: {
  citation: Citation | null;
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const isAttachment = citation?.type === "attachment";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-lg overflow-y-auto p-0">
        {citation && (
          <div className="flex flex-col">
            <SheetHeader className="border-b border-border p-6">
              <div className="flex items-center gap-2 text-xs text-ink-muted">
                {isAttachment ? (
                  <>
                    <FileText className="h-3.5 w-3.5" /> Attachment
                  </>
                ) : (
                  <>
                    <Mail className="h-3.5 w-3.5" /> Email message
                  </>
                )}
                {citation.threadLabel && (
                  <>
                    <span>·</span>
                    <span className="truncate">{citation.threadLabel}</span>
                  </>
                )}
              </div>
              <SheetTitle className="mt-1 text-lg leading-snug text-ink">
                {citation.subject ?? (isAttachment ? "Cited attachment page" : "Cited message")}
              </SheetTitle>
              {/* Sender and date only appear when the backend returned them. */}
              {(citation.sender || citation.date) && (
                <SheetDescription className="flex flex-wrap items-center gap-x-2 text-xs">
                  {citation.sender && (
                    <>
                      <User className="h-3.5 w-3.5" />
                      <span className="font-medium text-ink">{citation.sender}</span>
                    </>
                  )}
                  {citation.sender && citation.date && <span>·</span>}
                  {citation.date && <span>{formatDate(citation.date)}</span>}
                </SheetDescription>
              )}
            </SheetHeader>

            <div className="space-y-5 p-6">
              {citation.attachment && (
                <div className="flex flex-wrap items-center gap-2">
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

              {citation.attachment?.ocr && (
                <p className="rounded-lg border border-warning/30 bg-warning/[0.06] px-4 py-3 text-xs leading-relaxed text-ink-muted">
                  This quote was read from a scanned page by OCR, so it may differ slightly from the
                  characters printed in the original document.
                </p>
              )}

              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
                  Exact evidence
                </div>
                <blockquote
                  className={`rounded-lg border-l-2 border-brand bg-surface px-4 py-3 text-sm text-ink ${isAttachment ? "font-mono" : "italic"}`}
                >
                  &ldquo;{citation.quote}&rdquo;
                </blockquote>
              </div>

              {citation.context && (
                <div>
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
                    Adjacent context
                  </div>
                  <p className="rounded-lg bg-muted px-4 py-3 text-sm leading-relaxed text-ink/85 whitespace-pre-wrap">
                    {citation.context}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
