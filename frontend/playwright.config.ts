import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against the REAL backend (seeded demo users, LLM off) and the Vite dev server.
 * The edge-fallback spec then cuts the browser off from /api/v1/triage, so the ONNX model and the
 * client-side safety rules run in an actual browser, exactly as they would during a backend outage.
 *   npx playwright install --with-deps chromium && npm run e2e
 */
const PY = process.env.PYTHON
  ?? (process.platform === "win32" ? `"${path.resolve(process.cwd(), "..", "venv", "Scripts", "python.exe")}"` : "python");
const PASSWORD = process.env.E2E_PASSWORD ?? "e2e-password-123456";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: "http://127.0.0.1:5173", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `${PY} -m uvicorn backend.app.main:app --port 8000`,
      cwd: "..",
      url: "http://127.0.0.1:8000/healthz",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        DATABASE_URL: "sqlite:///./e2e.db", LLM_ENABLED: "false", SEED_ON_STARTUP: "true", SEED_DEMO_USERS: "true",
        DEMO_USER_PASSWORD: PASSWORD, JWT_SECRET: "e2e-secret-e2e-secret-e2e-secret-123456", RATE_LIMIT_ENABLED: "false",
        EMBEDDING_MODEL: "hash-v2", STT_ENGINE: "none", OCR_ENGINE: "none", CORS_ORIGINS: '["http://127.0.0.1:5173"]',
      },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5173 --strictPort",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: { VITE_API_URL: "http://127.0.0.1:8000" },
    },
  ],
});
