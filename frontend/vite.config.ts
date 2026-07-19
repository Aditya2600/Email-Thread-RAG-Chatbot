// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - TanStack devtools (dev-only, first), tanstackStart, viteReact, tailwindcss, tsConfigPaths,
//     nitro (build-only using cloudflare as a default target), VITE_* env injection, @ path alias,
//     React/TanStack dedupe, error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import { loadEnv } from "vite";

// The FastAPI backend serves its routes at the origin root, so each one is
// proxied by name. Proxying "/" wholesale would swallow the app's own routes.
const BACKEND_ROUTES = [
  "/health",
  "/threads",
  "/start_session",
  "/switch_thread",
  "/reset_session",
  "/ask",
  "/gmail",
  "/openapi.json",
];

// Read here rather than from import.meta.env: the proxy target is dev-server
// configuration, not something the browser bundle needs.
const backendOrigin =
  loadEnv("development", process.cwd(), "VITE_").VITE_BACKEND_ORIGIN ?? "http://localhost:8000";

export default defineConfig({
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    // nitro/vite builds from this
    server: { entry: "server" },
  },
  vite: {
    server: {
      // Dev-only. The browser calls relative paths, so no CORS policy is needed
      // on the backend. In production the app and the API must share an origin
      // behind one reverse proxy, or the backend must allow this origin itself.
      proxy: Object.fromEntries(
        BACKEND_ROUTES.map((route) => [route, { target: backendOrigin, changeOrigin: true }]),
      ),
    },
  },
});
