import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ask": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/knowledge-bases": "http://127.0.0.1:8000",
      "/ready": "http://127.0.0.1:8000",
      "/warmup": "http://127.0.0.1:8000"
    }
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts"
  }
});
