import { createFileRoute } from "@tanstack/react-router";
import { PageShell } from "@/components/app/PageShell";
import { Unavailable } from "@/components/app/Unavailable";

export const Route = createFileRoute("/sync")({
  component: SyncPage,
  head: () => ({
    meta: [
      { title: "Sync Activity · Inbox Copilot" },
      { name: "description", content: "Gmail sync activity for Inbox Copilot." },
    ],
  }),
});

// Gmail sync runs server-side off Pub/Sub pushes. The backend keeps that history
// in its own store and publishes no route for it, so there is nothing to render.
function SyncPage() {
  return (
    <PageShell
      title="Sync Activity"
      description="What Inbox Copilot has pulled in from your mailbox."
    >
      <Unavailable
        title="Sync history isn't available yet"
        endpoint="a sync-history route on the API"
      >
        Gmail sync runs in the background from Pub/Sub notifications, but the backend doesn't
        publish its history. Once it exposes a route, this timeline will show real events rather
        than a sample.
      </Unavailable>
    </PageShell>
  );
}
