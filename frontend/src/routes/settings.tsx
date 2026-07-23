import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Mail, Shield, Sparkles, Loader2 } from "lucide-react";
import { PageShell } from "@/components/app/PageShell";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api";
import { setAccountEmail } from "@/lib/account";
import { useGmailAvailability, useGmailConnect } from "@/lib/queries";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
  // The OAuth callback lands the browser back here as ?gmail=connected&email=…
  validateSearch: (search: Record<string, unknown>) => ({
    gmail: search.gmail === "connected" ? ("connected" as const) : undefined,
    email: typeof search.email === "string" ? search.email : undefined,
  }),
  head: () => ({
    meta: [
      { title: "Settings · Inbox Copilot" },
      {
        name: "description",
        content: "Connect a mailbox and review how Inbox Copilot handles your messages.",
      },
    ],
  }),
});

function SettingsPage() {
  return (
    <PageShell
      title="Settings"
      description="Connect a mailbox and review how Inbox Copilot handles your messages."
    >
      <div className="grid gap-5">
        <GmailSection />

        <section className="glass-card rounded-2xl p-5">
          <div className="flex items-start gap-3">
            <Shield className="mt-0.5 h-4 w-4 text-brand" />
            <div>
              <h2 className="text-sm font-semibold text-ink">Privacy & citations</h2>
              <p className="mt-1 text-sm text-ink-muted">
                Every answer is anchored to exact quoted evidence from your own mailbox. Message
                text stays inside your workspace, and the browser never handles OAuth secrets: the
                backend performs the token exchange and returns only the authorization URL.
              </p>
              <div className="mt-3 inline-flex items-center gap-1.5 text-xs text-ink-muted">
                <Sparkles className="h-3 w-3 text-brand" />
                No answer without a citation.
              </div>
            </div>
          </div>
        </section>
      </div>
    </PageShell>
  );
}

/**
 * The backend mounts its Gmail OAuth routes only when it has a Pub/Sub
 * subscription and a database configured. When they're not, this section is
 * simply absent rather than announcing what's missing.
 */
function GmailSection() {
  const gmail = useGmailAvailability();
  const connect = useGmailConnect();
  const { gmail: connected, email } = Route.useSearch();
  const [tenantId, setTenantId] = useState("");
  const [mailboxId, setMailboxId] = useState("");

  // The callback carries the real connected address; remember it for the sidebar.
  useEffect(() => {
    if (connected && email) setAccountEmail(email);
  }, [connected, email]);

  if (!gmail.data?.oauthAvailable) return null;

  const ready = tenantId.trim() !== "" && mailboxId.trim() !== "";

  return (
    <section className="glass-card rounded-2xl p-5">
      <div className="flex items-start gap-4">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-brand text-white">
          <Mail className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-ink">Gmail connection</h2>
          <p className="mt-1 text-sm text-ink-muted">
            Connecting opens Google's consent screen. Inbox Copilot requests read-only access to
            messages and attachments; the backend holds the credentials.
          </p>
          {connected && (
            <p className="mt-3 rounded-lg bg-brand/10 px-3 py-2 text-sm text-ink">
              Connected{email ? ` ${email}` : ""}. Inbox Copilot is now syncing this mailbox.
            </p>
          )}
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <Field label="Tenant ID" value={tenantId} onChange={setTenantId} placeholder="acme" />
            <Field
              label="Mailbox ID"
              value={mailboxId}
              onChange={setMailboxId}
              placeholder="jordan@acme.com"
            />
          </div>
          <Button
            size="sm"
            className="mt-4 gap-1.5"
            disabled={!ready || connect.isPending}
            onClick={() =>
              connect.mutate({ tenantId: tenantId.trim(), mailboxId: mailboxId.trim() })
            }
          >
            {connect.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {connect.isPending ? "Opening Google…" : "Connect Gmail"}
          </Button>
          {connect.isError && (
            <p className="mt-2 text-sm text-destructive">
              {connect.error instanceof ApiError
                ? connect.error.message
                : "Couldn't start the Gmail authorization."}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  return (
    <label className="block">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
        {label}
      </span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="mt-1 h-10 w-full rounded-lg border border-border bg-surface px-3 text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-ring/40"
      />
    </label>
  );
}
