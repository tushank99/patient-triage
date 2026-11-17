/* eslint-disable prettier/prettier */
/**
 * Edge fallback: the same XGBoost model, run locally via ONNX + WebAssembly, used ONLY when the
 * backend is unreachable or returns 5xx (see shouldUseEdgeFallback in api.ts).
 *
 * Everything model-specific comes from the bundle `python -m backend.training.promote` copies here:
 *   /models/triage_model.onnx       the model
 *   /models/triage_model.meta.json  feature order, derived features, class_to_esi, calibration temperature
 *   /models/safety_rules.json       the SAME deterministic rules the backend applies
 *
 * Missing vitals are sent as NaN plus explicit *_missing indicators, never imputed as normal. The maths that must
 * equal the server's lives in edge-math.ts and is checked in CI by scripts/verify-onnx-parity.mjs.
 */
import * as ort from "onnxruntime-web";
import type { Badge, AssessmentView } from "./assessment-view";
import { buildFeatureVector, decode, type ModelMeta } from "./edge-math";

// The ORT runtime (.mjs + .wasm) is served from /public. Importing a /public .mjs directly is refused by the Vite dev
// server, so the module is loaded through a Blob URL and the .wasm by absolute URL: works in dev and in production.
let runtimeReady: Promise<void> | null = null;
function prepareRuntime(): Promise<void> {
  if (!runtimeReady) {
    runtimeReady = (async () => {
      const name = "ort-wasm-simd-threaded.jsep";
      const res = await fetch(`/${name}.mjs`);
      if (!res.ok) throw new Error(`Failed to load ORT runtime (${res.status})`);
      const mjs = URL.createObjectURL(new Blob([await res.text()], { type: "text/javascript" }));
      ort.env.wasm.wasmPaths = { mjs, wasm: new URL(`/${name}.wasm`, window.location.origin).href };
    })().catch((e) => { runtimeReady = null; throw e; });
  }
  return runtimeReady;
}

const BUNDLE = "/models";

interface Condition { field: string; op: string; value: unknown }
interface SafetyRule { id: string; version: number; esi: number; reason: string; source: string; all?: Condition[]; any?: Condition[] }
interface RuleSet { ruleset_version: string; rules: SafetyRule[] }

/** Field names match the backend feature schema / VitalSigns. */
export interface EdgeModelInput {
  age_years: number;
  heart_rate?: number | null;
  systolic_bp?: number | null;
  diastolic_bp?: number | null;
  respiratory_rate?: number | null;
  oxygen_saturation?: number | null;
  temperature_celsius?: number | null;
  pain_score?: number | null;
  has_prior_history?: boolean;
  complaint?: string;
}

let bundlePromise: Promise<{ session: ort.InferenceSession; meta: ModelMeta; rules: RuleSet }> | null = null;
let rulesPromise: Promise<RuleSet> | null = null;

/** The safety rules are plain JSON: they must keep working even when the ONNX runtime cannot load. */
function loadRules(): Promise<RuleSet> {
  if (!rulesPromise) {
    rulesPromise = fetchJson<RuleSet>(`${BUNDLE}/safety_rules.json`).catch((e) => { rulesPromise = null; throw e; });
  }
  return rulesPromise;
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path} (${res.status})`);
  return (await res.json()) as T;
}

function loadBundle() {
  if (!bundlePromise) {
    bundlePromise = (async () => {
      await prepareRuntime();
      const [meta, rules, session] = await Promise.all([
        fetchJson<ModelMeta>(`${BUNDLE}/triage_model.meta.json`),
        loadRules(),
        ort.InferenceSession.create(`${BUNDLE}/triage_model.onnx`),
      ]);
      return { session, meta, rules };
    })().catch((e) => {
      bundlePromise = null; // allow a retry later
      throw e;
    });
  }
  return bundlePromise;
}

function conditionHolds(inputs: Record<string, unknown>, c: Condition): boolean {
  const actual = c.field.split(".").reduce<unknown>(
    (cur, key) => (cur !== null && typeof cur === "object" ? (cur as Record<string, unknown>)[key] : undefined),
    inputs,
  );
  if (actual === null || actual === undefined) return false; // never fire on data not collected
  const expected = c.value;
  switch (c.op) {
    case "<": return (actual as number) < (expected as number);
    case "<=": return (actual as number) <= (expected as number);
    case ">": return (actual as number) > (expected as number);
    case ">=": return (actual as number) >= (expected as number);
    case "==": return actual === expected;
    case "contains_any": {
      const text = String(actual).toLowerCase();
      return (expected as string[]).some((p) => text.includes(p.toLowerCase()));
    }
    case "has_word_any": {
      const text = String(actual).toLowerCase();
      return (expected as string[]).some((w) =>
        new RegExp(`\\b${w.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`).test(text),
      );
    }
    default:
      throw new Error(`Unknown rule operator ${c.op}`);
  }
}

/** Same semantics as backend/app/clinical/rules_engine.py. Server-only rules (history.*) never fire here. */
export function evaluateRules(rules: RuleSet, inputs: Record<string, unknown>): SafetyRule[] {
  return rules.rules.filter(
    (r) => (r.all ?? []).every((c) => conditionHolds(inputs, c)) && (!r.any || r.any.some((c) => conditionHolds(inputs, c))),
  );
}

/** Rules-only guidance when the ONNX model cannot run: a rule-flagged patient is still escalated. */
async function rulesOnlyAssessment(input: EdgeModelInput): Promise<AssessmentView> {
  const rules = await loadRules();
  const fired = evaluateRules(rules, { ...input, complaint: input.complaint ?? "" });
  const floor = fired.length ? Math.min(...fired.map((r) => r.esi)) : null;
  const esi = floor ?? 3;
  const reasons = fired.filter((r) => r.esi === floor).map((r) => r.reason).join("; ");
  return {
    source: "edge",
    modelNote: `safety rules ${rules.ruleset_version} only (offline model failed to load)`,
    recommended_esi: esi,
    confidence: { model_probability: null, data_completeness: 0, retrieval_confidence: null, retrieval_note: "Offline: no patient history available.", missing_critical: [] },
    rationale: floor !== null
      ? `SAFETY RULE ESCALATION to ESI ${esi} (${reasons}). Offline assessment: the local model is unavailable; rules only.`
      : "Offline assessment: the local model is unavailable and no safety rule fired. Defaulting to ESI 3 pending manual assessment.",
    riskFactors: fired.map((r) => r.reason),
    badges: [{ label: "Safety rules only (offline)", type: "danger" }, { label: "Cloud AI unavailable", type: "danger" }],
    rulesFired: fired.map(({ id, version, esi: e, reason, source }) => ({ id, version, esi: e, reason, source })),
    history: [],
    hasHistory: false,
    degraded: ["backend_unreachable", "edge_model_unavailable"],
    warnings: [],
    reviewReasons: ["Offline guidance only: nothing was saved. Assess manually and re-submit when the connection returns."],
    requiresConfirmation: false,
  };
}

export async function runEdgeInference(input: EdgeModelInput): Promise<AssessmentView> {
  let bundle;
  try {
    bundle = await loadBundle();
  } catch (e) {
    console.error("Edge ONNX model unavailable, falling back to safety rules only:", e);
    return rulesOnlyAssessment(input);
  }
  const { session, meta, rules } = bundle;
  const inputName = session.inputNames[0];
  if (!inputName) throw new Error("ONNX model has no inputs");

  const tensor = new ort.Tensor("float32", buildFeatureVector(meta, input as unknown as Record<string, number | boolean | null | undefined>), [1, meta.features.length]);
  const results = await session.run({ [inputName]: tensor });

  // Decode from the probabilities tensor via class_to_esi. The raw "label" output is a CLASS INDEX
  // (0 = ESI 1); the old code read it as an ESI level, shifting every prediction by one.
  const probOutput = results["probabilities"];
  if (!probOutput) throw new Error("ONNX model has no 'probabilities' output");
  const decoded = decode(meta, probOutput.data as Float32Array);  // calibration + class_to_esi, identical to the server
  const probs = decoded.probs;
  const best = probs.indexOf(decoded.top);
  const modelEsi = decoded.esi;

  const fired = evaluateRules(rules, { ...input, complaint: input.complaint ?? "" });
  const floor = fired.length ? Math.min(...fired.map((r) => r.esi)) : null;
  const finalEsi = floor !== null ? Math.min(modelEsi, floor) : modelEsi;

  const vitalWeights: [number | null | undefined, number][] = [
    [input.oxygen_saturation, 0.25], [input.heart_rate, 0.2], [input.systolic_bp, 0.2],
    [input.respiratory_rate, 0.15], [input.temperature_celsius, 0.1], [input.pain_score, 0.1],
  ];
  const completeness = Math.round(vitalWeights.reduce((sum, [v, w]) => sum + (v != null ? w : 0), 0) * 1000) / 1000;

  const badges: Badge[] = [
    { label: "Edge ONNX model (offline)", type: "warning" },
    { label: "Cloud AI unavailable", type: "danger" },
  ];
  const escalation = floor !== null && floor < modelEsi
    ? `SAFETY RULE ESCALATION to ESI ${finalEsi} (${fired.filter((r) => r.esi === floor).map((r) => r.reason).join("; ")}). `
    : "";

  return {
    source: "edge",
    modelNote: `${meta.model_version} (offline copy) · labels: ${meta.label_source ?? "unknown"}`,
    recommended_esi: finalEsi,
    confidence: {
      model_probability: probs[best] ?? null,
      data_completeness: completeness,
      retrieval_confidence: null,
      retrieval_note: "Offline: no patient history available.",
      missing_critical: [],
    },
    rationale: `${escalation}Offline assessment: model ${meta.model_version} suggested ESI ${modelEsi}. No history or AI narrative available offline.`,
    riskFactors: fired.map((r) => r.reason),
    badges,
    rulesFired: fired.map(({ id, version, esi, reason, source }) => ({ id, version, esi, reason, source })),
    history: [],
    hasHistory: false,
    degraded: ["backend_unreachable"],
    warnings: [],
    reviewReasons: ["Offline guidance only: nothing was saved. Re-submit when the connection returns."],
    requiresConfirmation: false,
  };
}
