import { createFileRoute } from "@tanstack/react-router";
import { CheckCircle2, Clock, Loader2, RefreshCw, XCircle } from "lucide-react";
import { PageShell } from "@/components/app/PageShell";
import { Unavailable } from "@/components/app/Unavailable";
import { ApiError, type SyncEvent } from "@/lib/api";
import { useSyncHistory } from "@/lib/queries";

export const Route = createFileRoute("/sync")({
  component: SyncPage,
  head: () => ({
    meta: [
      { title: "Sync Activity · Inbox Copilot" },
      { name: "description", content: "Gmail sync activity for Inbox Copilot." },
    ],
  }),
});

// Gmail sync runs server-side off Pub/Sub pushes. Each row in gmail_sync_jobs is
// one push-driven sync; GET /gmail/sync-history exposes them newest-first.
function SyncPage() {
  const { data, error, isLoading } = useSyncHistory();

  // 404 => the backend build has no sync-history route (Gmail not configured).
  if (error instanceof ApiError && error.status === 404) {
    return (
      <Shell>
        <Unavailable title="Sync history isn't available yet" endpoint="GET /gmail/sync-history">
          This backend build doesn't expose the sync-history route. Connect a mailbox and enable the
          Gmail integration to see real events here.
        </Unavailable>
      </Shell>
    );
  }

  return (
    <Shell>
      {isLoading && (
        <p className="text-sm text-ink-muted">
          <Loader2 className="mr-2 inline h-3.5 w-3.5 animate-spin" />
          Loading sync activity…
        </p>
      )}
      {error && !(error instanceof ApiError && error.status === 404) && (
        <p className="text-sm text-destructive">Couldn't load sync activity.</p>
      )}
      {data && data.length === 0 && (
        <p className="text-sm text-ink-muted">
          No syncs yet. New mail arrives here as Gmail pushes notifications.
        </p>
      )}
      {data && data.length > 0 && (
        <ol className="space-y-2">
          {data.map((event) => (
            <SyncRow key={event.id} event={event} />
          ))}
        </ol>
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <PageShell
      title="Sync Activity"
      description="What Inbox Copilot has pulled in from your mailbox."
    >
      {children}
    </PageShell>
  );
}

const STATUS: Record<
  SyncEvent["status"],
  { icon: typeof CheckCircle2; label: string; className: string }
> = {
  done: { icon: CheckCircle2, label: "Synced", className: "text-emerald-500" },
  running: { icon: RefreshCw, label: "Running", className: "text-brand" },
  pending: { icon: Clock, label: "Queued", className: "text-ink-muted" },
  failed: { icon: XCircle, label: "Failed", className: "text-destructive" },
};

function SyncRow({ event }: { event: SyncEvent }) {
  const status = STATUS[event.status];
  const Icon = status.icon;
  const when = new Date(event.completedAt ?? event.createdAt);
  return (
    <li className="glass-card flex items-start gap-3 rounded-xl p-3.5">
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${status.className}`} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-ink">{status.label}</span>
          <span className="truncate text-xs text-ink-muted">{event.mailboxId}</span>
        </div>
        {event.lastError && (
          <p className="mt-1 truncate text-xs text-destructive">{event.lastError}</p>
        )}
        {event.needsFullSync && (
          <p className="mt-1 text-xs text-ink-muted">Full resync</p>
        )}
      </div>
      <time className="shrink-0 text-xs text-ink-muted" dateTime={when.toISOString()}>
        {when.toLocaleString()}
      </time>
    </li>
  );
}
