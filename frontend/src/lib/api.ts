/* eslint-disable prettier/prettier */
/**
 * The only place the frontend talks to the backend.
 *
 * Request/response types come from api-types.ts, which is GENERATED from contracts/openapi.json
 * (`npm run gen:api`). If the backend contract changes, regenerating makes `tsc` fail anywhere this
 * file or its callers build the wrong shape.
 *
 * Auth: the bearer token lives in sessionStorage (cleared when the tab closes) and is attached to every
 * request. A 401 anywhere signs the user out. The browser never sends a clinician id: the backend takes
 * the actor from the token.
 */
import type { components } from "./api-types";

type Schemas = components["schemas"];
export type TriageRequest = Schemas["TriageRequest"];
export type ReassessRequest = Schemas["ReassessRequest"];
export type TriageResponse = Schemas["TriageResponse"];
export type VitalSigns = Schemas["VitalSigns"];
export type PatientLookup = Schemas["PatientLookup"];
export type OverrideRequest = Schemas["OverrideRequest"];
export type OverrideResponse = Schemas["OverrideResponse"];
export type HistoryDocumentRef = Schemas["HistoryDocumentRef"];
export type RuleHit = Schemas["RuleHit"];
export type ConfidenceBreakdown = Schemas["ConfidenceBreakdown"];
export type QueueItem = Schemas["QueueItem"];
export type ArchiveItem = Schemas["ArchiveItem"];
export type VitalsReadingOut = Schemas["VitalsReadingOut"];
export type AuditEventOut = Schemas["AuditEventOut"];
export type UserOut = Schemas["UserOut"];
export type TokenResponse = Schemas["TokenResponse"];
export type Disposition = Schemas["DispositionRequest"]["disposition"];
export type DocumentOut = Schemas["DocumentOut"];
export type DocumentDetail = Schemas["DocumentDetail"];
export type ChunkOut = Schemas["ChunkOut"];
export type UploadResponse = Schemas["UploadResponse"];
export const DOC_TYPES = [
  "discharge_summary", "er_note", "clinic_note", "lab_summary", "imaging_report", "immunization_record",
  "medication_list", "allergy_record", "problem_list", "other",
] as const;

export const API_URL: string = import.meta.env["VITE_API_URL"] || "http://127.0.0.1:8000";

/* ---------------------------------------------------------------- token store */

const TOKEN_KEY = "triageflow.token";
let token: string | null = null;
let onUnauthorized: (() => void) | null = null;

try { token = sessionStorage.getItem(TOKEN_KEY); } catch { /* storage unavailable */ }

export function setToken(value: string | null): void {
  token = value;
  try {
    if (value) sessionStorage.setItem(TOKEN_KEY, value); else sessionStorage.removeItem(TOKEN_KEY);
  } catch { /* storage unavailable */ }
}
export const getToken = (): string | null => token;
export function setUnauthorizedHandler(fn: (() => void) | null): void { onUnauthorized = fn; }

/* ---------------------------------------------------------------- errors */

/** HTTP error with the backend's standard envelope: {"error": {code, message, details}}. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly details: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** 4xx = our request was wrong (contract bug, unknown MRN, not permitted). Show it; do NOT fall back. */
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }
}

/**
 * True when the backend is unreachable or broken (network error, timeout, 5xx).
 * Only these may trigger the offline ONNX guidance. A 4xx must never be masked by it.
 */
export function shouldUseEdgeFallback(err: unknown): boolean {
  return !(err instanceof ApiError) || !err.isClientError;
}

export function newRequestId(): string {
  return crypto.randomUUID();
}

export function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.code === "validation_error") return `The request did not match the API contract: ${err.message}`;
    return err.message;
  }
  return "Could not reach the triage service.";
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 25000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),  // browser sets the multipart boundary
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init?.headers ?? {}),
      },
      signal: controller.signal,
    });
    if (!res.ok) {
      let code = "http_error";
      let message = `Request failed (${res.status})`;
      let details: unknown = null;
      try {
        const body = (await res.json()) as { error?: { code?: string; message?: string; details?: unknown } };
        code = body.error?.code ?? code;
        message = body.error?.message ?? message;
        details = body.error?.details ?? null;
      } catch { /* non-JSON error body */ }
      if (res.status === 401 && path !== "/api/v1/auth/login") onUnauthorized?.();
      throw new ApiError(res.status, code, message, details);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

const post = <T>(path: string, body: unknown = {}) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

/* ---------------------------------------------------------------- endpoints */

export const login = (username: string, password: string) => post<TokenResponse>("/api/v1/auth/login", { username, password });
export const fetchMe = () => request<UserOut>("/api/v1/auth/me");

export const triagePatient = (body: TriageRequest) => post<TriageResponse>("/api/v1/triage", body);
export const reassessVisit = (visitId: string, body: ReassessRequest) =>
  post<TriageResponse>(`/api/v1/visits/${encodeURIComponent(visitId)}/reassess`, body);
export const overrideEsi = (body: OverrideRequest) => post<OverrideResponse>("/api/v1/override", body);
export const lookupPatient = (mrn: string) => request<PatientLookup>(`/api/v1/patients/${encodeURIComponent(mrn.trim())}`);

export const fetchQueue = () => request<QueueItem[]>("/api/v1/queue");
export const fetchArchive = () => request<ArchiveItem[]>("/api/v1/archive");
export const dispositionVisit = (visit_id: string, disposition: Disposition, checklist?: Record<string, boolean>) =>
  post("/api/v1/archive", { visit_id, disposition, checklist: checklist ?? null });
export const revertDisposition = (visit_id: string) => post("/api/v1/archive/revert", { visit_id });
export const anonymizePatient = (mrn: string) => post("/api/v1/archive/anonymize", { mrn });
export const acknowledgeReview = (visitId: string, note?: string) =>
  post(`/api/v1/visits/${encodeURIComponent(visitId)}/review`, { note: note ?? null });
export const fetchVitalsHistory = (visitId: string) => request<VitalsReadingOut[]>(`/api/v1/visits/${encodeURIComponent(visitId)}/vitals`);
export const fetchAudit = (visitId: string) => request<AuditEventOut[]>(`/api/v1/visits/${encodeURIComponent(visitId)}/audit`);
export const startSurge = () => post<{ visits_changed: number }>("/api/v1/demo/surge");
export const endSurge = () => post<{ visits_changed: number }>("/api/v1/demo/surge/end");

export const listDocuments = (mrn: string) => request<DocumentOut[]>(`/api/v1/patients/${encodeURIComponent(mrn)}/documents`);
export const fetchDocument = (id: string) => request<DocumentDetail>(`/api/v1/documents/${encodeURIComponent(id)}`);
export const fetchDocumentChunks = (id: string) => request<ChunkOut[]>(`/api/v1/documents/${encodeURIComponent(id)}/chunks`);
export const deleteDocument = (id: string) => request(`/api/v1/documents/${encodeURIComponent(id)}`, { method: "DELETE" });

export function uploadDocument(mrn: string, file: File, docType: string, title?: string, docDate?: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("doc_type", docType);
  if (title) form.append("title", title);
  if (docDate) form.append("doc_date", docDate);
  return request<UploadResponse>(`/api/v1/patients/${encodeURIComponent(mrn)}/documents`, { method: "POST", body: form }, 120000);
}

export function uploadDocumentVersion(documentId: string, file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResponse>(`/api/v1/documents/${encodeURIComponent(documentId)}/versions`, { method: "POST", body: form }, 120000);
}

export type CaptureOut = Schemas["CaptureOut"];

export function transcribeAudio(audio: Blob, requestId: string, mrn?: string, language?: string): Promise<CaptureOut> {
  const form = new FormData();
  form.append("file", audio, "dictation" + (audio.type.includes("webm") ? ".webm" : audio.type.includes("ogg") ? ".ogg" : ".wav"));
  form.append("request_id", requestId);
  if (mrn) form.append("mrn", mrn);
  if (language) form.append("language", language);
  return request<CaptureOut>("/api/v1/intake/transcribe", { method: "POST", body: form }, 180000);
}
export const verifyCapture = (id: string, text?: string, fields?: Record<string, unknown>[]) =>
  post<CaptureOut>(`/api/v1/captures/${encodeURIComponent(id)}/verify`, { text: text ?? null, fields: fields ?? null });
export const rejectCapture = (id: string, reason: string) => post<CaptureOut>(`/api/v1/captures/${encodeURIComponent(id)}/reject`, { reason });
export const listCaptures = (mrn: string, status?: string) =>
  request<CaptureOut[]>(`/api/v1/patients/${encodeURIComponent(mrn)}/captures${status ? `?status=${status}` : ""}`);

/** Parse an optional numeric form field; empty or invalid -> null (never 0). */
export function optionalNumber(value: string, integer = false): number | null {
  if (value.trim() === "") return null;
  const n = integer ? parseInt(value, 10) : parseFloat(value);
  return Number.isFinite(n) ? n : null;
}
