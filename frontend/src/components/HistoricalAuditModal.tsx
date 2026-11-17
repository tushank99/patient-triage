/* eslint-disable prettier/prettier */
import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { ArchivedPatient, ESI_META } from "../lib/triage-context";
import { Activity, BrainCircuit, UserCog, UserCheck, ShieldAlert, ClipboardCheck, AlertTriangle } from "lucide-react";
import { describeError, fetchAudit, fetchVitalsHistory, type AuditEventOut, type VitalsReadingOut } from "../lib/api";
import { useAuth } from "../lib/auth";

interface Props {
  patient: ArchivedPatient | null;
  isOpen: boolean;
  onClose: () => void;
}

const time = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const ICON: Record<string, { icon: React.ReactNode; tone: string; title: string }> = {
  AI_TRIAGE: { icon: <BrainCircuit className="w-4 h-4" />, tone: "bg-emerald-100 text-emerald-600", title: "AI Assessment" },
  AI_REASSESS: { icon: <Activity className="w-4 h-4" />, tone: "bg-blue-100 text-blue-600", title: "Re-triage with new vitals" },
  OVERRIDE_ESI: { icon: <UserCog className="w-4 h-4" />, tone: "bg-amber-100 text-amber-600", title: "Clinician Override" },
  OVERRIDE_ESI_BELOW_RULE_FLOOR: { icon: <AlertTriangle className="w-4 h-4" />, tone: "bg-red-100 text-red-600", title: "Override BELOW safety-rule floor" },
  REVIEW_ACKNOWLEDGED: { icon: <ClipboardCheck className="w-4 h-4" />, tone: "bg-blue-100 text-blue-600", title: "Review acknowledged" },
  DETERIORATION_FLAGGED: { icon: <AlertTriangle className="w-4 h-4" />, tone: "bg-red-100 text-red-600", title: "Flagged for reassessment (wait limit)" },
  PATIENT_DISPOSITIONED: { icon: <UserCheck className="w-4 h-4" />, tone: "bg-slate-800 text-white", title: "Patient dispositioned" },
  DISPOSITION_REVERTED: { icon: <UserCheck className="w-4 h-4" />, tone: "bg-yellow-100 text-yellow-700", title: "Disposition reverted" },
};

export function HistoricalAuditModal({ patient, isOpen, onClose }: Props) {
  const { can } = useAuth();
  const allowed = can("patient:read");
  const [events, setEvents] = useState<AuditEventOut[]>([]);
  const [vitals, setVitals] = useState<VitalsReadingOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !patient || !allowed) return;
    setEvents([]); setVitals([]); setError(null);
    Promise.all([fetchAudit(patient.id), fetchVitalsHistory(patient.id)])
      .then(([a, v]) => { setEvents(a); setVitals(v); })
      .catch((err) => setError(describeError(err)));
  }, [isOpen, patient, allowed]);

  if (!patient) return null;
  const meta = ESI_META[patient.esi];

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl bg-slate-50 max-h-[90vh] overflow-y-auto">
        <DialogHeader className="pb-4 border-b border-slate-200">
          <div className="flex justify-between items-start">
            <div>
              <DialogTitle className="text-xl flex items-center gap-2">
                {patient.isAnonymized && <ShieldAlert className="w-5 h-5 text-red-500" />}
                Audit Trail: {patient.name}
              </DialogTitle>
              <p className="text-sm text-slate-500 mt-1">MRN: {patient.mrn}</p>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-bold shadow-sm ${meta.badge}`}>Final ESI: {patient.esi}</span>
          </div>
        </DialogHeader>

        {!allowed ? (
          <p className="py-6 text-sm text-slate-600">Your role does not have patient-level access, so the clinical audit trail is hidden.</p>
        ) : (
          <div className="py-4 space-y-5">
            <p className="text-sm text-slate-700 italic border-l-2 border-slate-300 pl-3">"{patient.complaint}"</p>
            {error && <p className="rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{error}</p>}

            {vitals.length > 0 && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h4 className="font-bold text-slate-900 mb-2 text-sm">Vitals history</h4>
                <table className="w-full text-xs">
                  <thead><tr className="text-left text-slate-500"><th>Time</th><th>HR</th><th>BP</th><th>RR</th><th>SpO2</th><th>Temp</th><th>Pain</th><th>ESI</th></tr></thead>
                  <tbody>
                    {vitals.map((v) => (
                      <tr key={v.recorded_at} className="border-t border-slate-100">
                        <td className="py-1">{time(v.recorded_at)} <span className="text-slate-400">{v.source === "REASSESS" ? "re" : ""}</span></td>
                        <td>{v.heart_rate ?? "—"}</td>
                        <td>{v.systolic_bp ? `${v.systolic_bp}/${v.diastolic_bp ?? "?"}` : "—"}</td>
                        <td>{v.respiratory_rate ?? "—"}</td>
                        <td>{v.oxygen_saturation ?? "—"}</td>
                        <td>{v.temperature_celsius ?? "—"}</td>
                        <td>{v.pain_score ?? "—"}</td>
                        <td className="font-bold">{v.ai_esi_after ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="space-y-3">
              {events.map((e, i) => {
                const m = ICON[e.action] ?? { icon: <Activity className="w-4 h-4" />, tone: "bg-slate-100 text-slate-600", title: e.action };
                const a = (e.detail as { assessment?: { components?: { rules_fired?: { id: string; reason: string }[]; llm_downgrade_blocked?: boolean }; confidence?: { model_probability?: number | null; data_completeness?: number }; decision_trace?: { stage: string; note: string }[] }; llm?: { provider: string; model: string; attempts: number; latency_ms: number; request_sha256: string } | null } | null) ?? null;
                return (
                  <div key={i} className="flex gap-3">
                    <div className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 shadow-sm ${m.tone}`}>{m.icon}</div>
                    <div className="bg-white p-3 rounded-lg shadow-sm border border-slate-200 flex-1 text-sm">
                      <div className="flex justify-between items-center">
                        <h4 className="font-bold text-slate-900">{m.title}</h4>
                        <span className="text-xs font-bold text-slate-500">{time(e.timestamp)} · {e.actor}</span>
                      </div>
                      <p className="text-xs text-slate-600 mt-1">
                        {e.initial_ai_esi != null && <>AI ESI {e.initial_ai_esi} · </>}Assigned ESI {e.assigned_esi}
                      </p>
                      {e.reason && <p className="text-xs text-slate-700 mt-1"><strong>Reason:</strong> {e.reason}</p>}
                      {a?.assessment && (
                        <div className="mt-2 space-y-1 rounded bg-slate-50 p-2 text-xs text-slate-600">
                          {a.assessment.confidence && (
                            <p>Model probability {a.assessment.confidence.model_probability != null ? `${Math.round(a.assessment.confidence.model_probability * 100)}%` : "n/a"} · data completeness {Math.round((a.assessment.confidence.data_completeness ?? 0) * 100)}%</p>
                          )}
                          {a.assessment.components?.rules_fired?.map((r) => <p key={r.id}>Rule {r.id}: {r.reason}</p>)}
                          {a.assessment.components?.llm_downgrade_blocked && <p className="font-semibold text-red-700">A safety rule overrode a less acute AI suggestion.</p>}
                          {a.assessment.decision_trace?.map((s, j) => <p key={j}>• {s.stage}: {s.note}</p>)}
                          {a.llm && <p>LLM {a.llm.provider}/{a.llm.model} · {a.llm.attempts} attempt(s) · {a.llm.latency_ms} ms · request {a.llm.request_sha256.slice(0, 10)}…</p>}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
