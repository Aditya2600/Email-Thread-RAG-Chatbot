import { CheckCircle2, AlertTriangle, Loader2, Mail, MailX } from "lucide-react";
import { useGmailAvailability, useHealth } from "@/lib/queries";

export function TopBar() {
  const health = useHealth();
  const gmail = useGmailAvailability();

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-end gap-2 border-b border-border bg-background/80 backdrop-blur-md px-4 md:px-6">
      <ApiStatusPill state={health.isPending ? "checking" : health.isError ? "down" : "up"} />

      {/* Gmail routes exist only when the backend is configured for them. */}
      {!gmail.isPending && (
        <div className="hidden sm:flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5">
          {gmail.data?.oauthAvailable ? (
            <>
              <Mail className="h-3.5 w-3.5 text-brand" />
              <span className="text-xs font-medium text-ink">Gmail available</span>
            </>
          ) : (
            <>
              <MailX className="h-3.5 w-3.5 text-ink-muted" />
              <span className="text-xs font-medium text-ink-muted">Gmail not configured</span>
            </>
          )}
        </div>
      )}
    </header>
  );
}

function ApiStatusPill({ state }: { state: "checking" | "up" | "down" }) {
  if (state === "checking") {
    return (
      <div className="flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" />
        <span className="text-xs font-medium text-ink-muted">Checking API</span>
      </div>
    );
  }
  if (state === "down") {
    return (
      <div className="flex items-center gap-2 rounded-full border border-destructive/30 bg-destructive/[0.06] px-3 py-1.5">
        <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
        <span className="text-xs font-medium text-destructive">API unreachable</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-60 animate-ping" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
      </span>
      <CheckCircle2 className="h-3.5 w-3.5 text-success" />
      <span className="text-xs font-medium text-ink">API connected</span>
    </div>
  );
}
