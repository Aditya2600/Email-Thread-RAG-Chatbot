import { Link, useRouterState } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Sparkles,
  FileText,
  Activity,
  Settings as SettingsIcon,
  Mail,
} from "lucide-react";
import { motion } from "framer-motion";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { getAccountEmail, initialsFromEmail } from "@/lib/account";

const nav: Array<{ to: string; label: string; icon: typeof Sparkles; exact?: boolean }> = [
  { to: "/", label: "Ask Inbox", icon: Sparkles, exact: true },
  { to: "/sources", label: "Sources", icon: FileText },
  { to: "/sync", label: "Sync Activity", icon: Activity },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

export function AppSidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  // Re-read on navigation so connecting a mailbox on /settings reflects here.
  const [email, setEmail] = useState<string | null>(null);
  useEffect(() => setEmail(getAccountEmail()), [pathname]);

  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-border bg-sidebar">
      <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-brand">
          <Sparkles className="h-4 w-4 text-white" strokeWidth={2.4} />
        </div>
        <div className="min-w-0 text-[15px] font-semibold tracking-tight text-ink">
          Inbox Copilot
        </div>
      </div>

      <nav className="flex-1 px-2 py-2">
        {nav.map((item) => {
          const active = item.exact ? pathname === item.to : pathname.startsWith(item.to);
          const Icon = item.icon;
          return (
            <Link
              key={item.to}
              to={item.to}
              className="relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-ink/80 hover:text-ink hover:bg-sidebar-accent transition-colors"
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-lg bg-sidebar-accent"
                  transition={{ type: "spring", stiffness: 400, damping: 34 }}
                />
              )}
              <Icon className={`relative h-4 w-4 ${active ? "text-brand" : "text-ink-muted"}`} />
              <span className={`relative ${active ? "font-medium text-ink" : ""}`}>
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border px-3 py-3">
        <div className="flex items-center gap-3 rounded-lg px-2 py-1.5">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-brand text-white text-xs">
              {email ? initialsFromEmail(email) : <Mail className="h-3.5 w-3.5" />}
            </AvatarFallback>
          </Avatar>
          <div className="min-w-0 flex-1">
            {email ? (
              <div className="flex items-center gap-1 text-[13px] text-ink">
                <Mail className="h-3 w-3 shrink-0 text-ink-muted" />
                <span className="truncate">{email}</span>
              </div>
            ) : (
              <Link
                to="/settings"
                search={{ gmail: undefined, email: undefined }}
                className="truncate text-sm text-ink-muted hover:text-ink"
              >
                No mailbox connected
              </Link>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
