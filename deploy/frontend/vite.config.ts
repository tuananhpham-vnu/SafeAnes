import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: the API runs on :8000 (uvicorn); production: nginx proxies /api (see nginx.conf)
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": process.env.SAFEANES_API ?? "http://localhost:8000" } },
});
