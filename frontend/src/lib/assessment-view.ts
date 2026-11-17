/* eslint-disable prettier/prettier */
import type { ConfidenceBreakdown, HistoryDocumentRef, RuleHit, TriageResponse } from "./api";

export type ModelDriver = TriageResponse["components"]["ml_drivers"][number];
export type Claim = TriageResponse["claims"][number];
export type Facts = TriageResponse["facts"];

export type Badge = TriageResponse["badges"][number];

/** What the triage screens render, whether the result came from the backend or the offline model. */
export interface AssessmentView {
  source: "cloud" | "edge" | "offline";
  recommended_esi: number;
  confidence: ConfidenceBreakdown;
  rationale: string;
  riskFactors: string[];
  badges: Badge[];
  rulesFired: RuleHit[];
  history: HistoryDocumentRef[];
  hasHistory: boolean;
  degraded: string[];
  warnings: string[];
  reviewReasons: string[];
  requiresConfirmation: boolean;
  visitId?: string | undefined;
  mrn?: string | undefined;
  finalEsi?: number | undefined;
  /** TreeSHAP drivers of the ML model (server only). */
  mlDrivers?: ModelDriver[] | undefined;
  /** e.g. 'xgb-v3-synthetic-rules-20261115 · labels: synthetic-rules' */
  modelNote?: string | undefined;
  /** No narrative note matched the complaint confidently; only pinned records were used. */
  historyLowConfidence?: boolean | undefined;
  retrievalMethod?: string | undefined;
  claims?: Claim[] | undefined;
  facts?: Facts | undefined;
  securityFlags?: string[] | undefined;
}

export function fromTriageResponse(r: TriageResponse): AssessmentView {
  return {
    source: "cloud",
    recommended_esi: r.recommended_esi,
    confidence: r.confidence,
    rationale: r.rationale,
    riskFactors: r.risk_factors,
    badges: r.badges,
    rulesFired: r.components.rules_fired,
    history: r.history.documents,
    hasHistory: r.history.has_history,
    degraded: r.degraded,
    warnings: r.warnings,
    reviewReasons: r.review_reasons,
    requiresConfirmation: r.requires_confirmation,
    visitId: r.visit_id,
    mrn: r.mrn,
    finalEsi: r.final_esi,
    mlDrivers: r.components.ml_drivers,
    historyLowConfidence: r.history.low_confidence,
    retrievalMethod: r.history.retrieval_method,
    claims: r.claims,
    facts: r.facts,
    securityFlags: r.security_flags,
    modelNote: r.versions.model ? `${r.versions.model} · labels: ${r.versions.model_label_source ?? "unknown"}` : undefined,
  };
}

/** Last resort when both the backend and the ONNX model fail. */
export function offlineDefault(): AssessmentView {
  return {
    source: "offline",
    recommended_esi: 3,
    confidence: { model_probability: null, data_completeness: 0, retrieval_confidence: null, retrieval_note: "Offline.", missing_critical: [] },
    rationale: "SYSTEM OFFLINE: cloud API and local model both unavailable. Defaulting to ESI 3 pending manual assessment.",
    riskFactors: [],
    badges: [{ label: "Fully offline", type: "danger" }],
    rulesFired: [],
    history: [],
    hasHistory: false,
    degraded: ["backend_unreachable", "edge_model_unavailable"],
    warnings: [],
    reviewReasons: ["No AI available: assess manually."],
    requiresConfirmation: false,
  };
}

export const pct = (x: number | null | undefined): string => (x == null ? "n/a" : `${Math.round(x * 100)}%`);
