import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Search, MessagesSquare } from "lucide-react";
import { PageShell } from "@/components/app/PageShell";
import { Unavailable } from "@/components/app/Unavailable";
import { useThreads } from "@/lib/queries";
import { Skeleton } from "@/components/ui/skeleton";

export const Route = createFileRoute("/sources")({
  component: SourcesPage,
  head: () => ({
    meta: [
      { title: "Sources · Inbox Copilot" },
      { name: "description", content: "The threads Inbox Copilot has indexed and can cite from." },
    ],
  }),
});

// The backend's only inventory route is GET /threads. It has no route that
// lists messages or attachments, so this page shows threads and says so.
function SourcesPage() {
  const threads = useThreads();
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (threads.data ?? []).filter((t) => !needle || t.toLowerCase().includes(needle));
  }, [threads.data, q]);

  return (
    <PageShell
      title="Sources"
      description="The threads Inbox Copilot has indexed. Ask a question to see the exact messages and attachment pages it cites."
    >
      {threads.isError ? (
        <Unavailable title="Can't reach the API" endpoint="GET /threads">
          Inbox Copilot's backend isn't responding, so the indexed threads can't be listed.
        </Unavailable>
      ) : threads.isPending ? (
        <div className="space-y-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-16 w-full rounded-xl" />
          ))}
        </div>
      ) : threads.data.length === 0 ? (
        <Unavailable title="Nothing indexed yet">
          The backend reports no indexed threads. Ingest a corpus or connect Gmail, then check back.
        </Unavailable>
      ) : (
        <>
          <div className="mb-5 relative max-w-md">
            <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-muted" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Filter threads…"
              className="w-full h-10 rounded-lg border border-border bg-surface pl-9 pr-3 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-ring/40"
            />
          </div>

          {filtered.length === 0 ? (
            <div className="glass-card rounded-xl p-10 text-center">
              <p className="text-sm text-ink">No threads match that filter</p>
              <p className="mt-1 text-xs text-ink-muted">
                Clear the search to see all indexed threads.
              </p>
            </div>
          ) : (
            <div className="grid gap-3">
              {filtered.map((thread) => (
                <div key={thread} className="glass-card rounded-xl p-4 lift-on-hover">
                  <div className="flex items-center gap-3">
                    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-brand text-white">
                      <MessagesSquare className="h-4 w-4" />
                    </div>
                    <span className="min-w-0 flex-1 truncate font-mono text-sm text-ink">
                      {thread}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          <p className="mt-5 text-xs text-ink-muted">
            Message- and attachment-level browsing isn't available: the backend exposes no route
            that lists them outside of an answer's citations.
          </p>
        </>
      )}
    </PageShell>
  );
}
