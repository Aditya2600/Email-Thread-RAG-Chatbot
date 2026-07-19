import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function PageShell({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="mx-auto w-full max-w-6xl px-4 md:px-8 py-8"
    >
      <div className="mb-6">
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1.5 text-sm text-ink-muted">{description}</p>}
      </div>
      {children}
    </motion.div>
  );
}
