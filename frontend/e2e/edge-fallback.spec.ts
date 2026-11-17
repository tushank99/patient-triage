import { expect, test } from "@playwright/test";

const PASSWORD = process.env.E2E_PASSWORD ?? "e2e-password-123456";

/**
 * Graceful degradation, for real: the triage API is unreachable, so the browser must run the bundled ONNX
 * model + client-side safety rules. A silent-hypoxia patient (SpO2 88 %) must still be escalated to ESI <= 2.
 */
test("ONNX edge fallback escalates hypoxia when the triage API is down", async ({ page }) => {
  await page.goto("/");
  await page.locator("#login-user").fill("lee");
  await page.locator("#login-pass").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText("Add Patient").first()).toBeVisible({ timeout: 30_000 });

  // Backend outage for triage only (login/queue keep working, like a crashed pipeline or DB-lock).
  let blocked = 0;
  await page.route("**/api/v1/triage", (route) => { blocked += 1; return route.abort("connectionrefused"); });

  await page.getByText("Add Patient").first().click();
  await page.locator("#patient-name").fill("Edge, Test");
  await page.locator("#patient-gender").selectOption({ index: 1 }).catch(() => undefined);
  await page.locator("#patient-dob").fill("1971-05-04");
  await page.locator("#patient-complaint").fill("Feels a bit winded walking up stairs, denies chest pain.");
  await page.getByPlaceholder("Sys").fill("122");
  await page.getByPlaceholder("Dia").fill("78");
  await page.locator("#vital-hr").fill("78");
  await page.locator("#vital-spo2").fill("88");
  await page.locator("#vital-rr").fill("24");
  await page.getByRole("button", { name: "Review Patient Details" }).click();
  await page.getByRole("button", { name: /Confirm & Run AI Triage/ }).click();

  await expect(page.getByText(/Offline assessment/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Edge ONNX model (offline)")).toBeVisible();
  await expect(page.getByText(/SAFETY RULE ESCALATION to ESI [12]/)).toBeVisible();
  expect(blocked).toBeGreaterThan(0);
});
