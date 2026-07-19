import type { ReactNode } from "react";
import { PlugZap } from "lucide-react";

/**
 * Shown where the UI has a place for data the backend has no route for. It says
 * so plainly instead of filling the space with sample history.
 */
export function Unavailable({
  title,
  children,
  endpoint,
}: {
  title: string;
  children: ReactNode;
  endpoint?: string;
}) {
  return (
    <div className="glass-card rounded-2xl p-10 text-center">
      <div className="mx-auto grid h-10 w-10 place-items-center rounded-lg bg-accent">
        <PlugZap className="h-4 w-4 text-brand" />
      </div>
      <h2 className="mt-4 text-sm font-semibold text-ink">{title}</h2>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-ink-muted">{children}</p>
      {endpoint && <p className="mt-3 font-mono text-[11px] text-ink-muted">Needs: {endpoint}</p>}
    </div>
  );
}
