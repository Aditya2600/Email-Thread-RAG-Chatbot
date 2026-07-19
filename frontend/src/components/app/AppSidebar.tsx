import { Link, useRouterState } from "@tanstack/react-router";
import {
  Sparkles,
  Inbox,
  FileText,
  Activity,
  Settings as SettingsIcon,
  ChevronDown,
  Mail,
} from "lucide-react";
import { motion } from "framer-motion";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

const nav: Array<{ to: string; label: string; icon: typeof Sparkles; exact?: boolean }> = [
  { to: "/", label: "Ask Inbox", icon: Sparkles, exact: true },
  { to: "/sources", label: "Sources", icon: FileText },
  { to: "/sync", label: "Sync Activity", icon: Activity },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

export function AppSidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-border bg-sidebar">
      <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-brand shadow-[var(--shadow-glow)]">
          <Sparkles className="h-4 w-4 text-white" strokeWidth={2.4} />
        </div>
        <div className="min-w-0">
          <div className="text-[15px] font-semibold tracking-tight text-ink">Inbox Copilot</div>
          <div className="text-[11px] text-ink-muted">Grounded email intelligence</div>
        </div>
      </div>

      <div className="px-3 pb-2">
        <DropdownMenu>
          <DropdownMenuTrigger className="group flex w-full items-center justify-between rounded-lg border border-border bg-surface-raised px-3 py-2 text-left text-sm text-ink hover:bg-accent transition-colors">
            <span className="flex items-center gap-2 min-w-0">
              <Inbox className="h-4 w-4 text-ink-muted shrink-0" />
              <span className="truncate">All mailboxes</span>
            </span>
            <ChevronDown className="h-4 w-4 text-ink-muted" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-56">
            <DropdownMenuItem>All mailboxes</DropdownMenuItem>
            <DropdownMenuItem>Inbox</DropdownMenuItem>
            <DropdownMenuItem>Project Atlas</DropdownMenuItem>
            <DropdownMenuItem>Finance</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
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
            <AvatarFallback className="bg-gradient-brand text-white text-xs">JO</AvatarFallback>
          </Avatar>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium text-ink">Jordan Okafor</div>
            <div className="flex items-center gap-1 text-[11px] text-ink-muted">
              <Mail className="h-3 w-3" />
              <span className="truncate">jordan@atlas-co.com</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
