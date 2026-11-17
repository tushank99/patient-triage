/* eslint-disable prettier/prettier */
import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { useTriage } from "../lib/triage-context";
import { runEdgeInference } from "../lib/onnx-engine";
import { describeError, newRequestId, optionalNumber, overrideEsi, reassessVisit, shouldUseEdgeFallback, type VitalSigns } from "../lib/api";
import { fromTriageResponse, type AssessmentView } from "../lib/assessment-view";

export function UpdateVitalsModal() {
  const { patients, updateVitalsModal, setUpdateVitalsModal, refresh } = useTriage();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offlineNote, setOfflineNote] = useState<string | null>(null);

  const [hr, setHr] = useState("");
  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [rr, setRr] = useState("");
  const [spo2, setSpo2] = useState("");
  const [temp, setTemp] = useState("");
  const [pain, setPain] = useState("");

  // One request id per attempt: re-submitting after a network error reuses it, so the server never double-applies.
  const [requestId, setRequestId] = useState(newRequestId);
  const [pending, setPending] = useState<AssessmentView | null>(null);

  const patient = patients.find((p) => p.id === updateVitalsModal.patientId);

  useEffect(() => {
    if (!updateVitalsModal.isOpen) {
      setHr(""); setBpSys(""); setBpDia(""); setRr(""); setSpo2(""); setTemp(""); setPain("");
      setPending(null); setError(null); setOfflineNote(null);
      setRequestId(newRequestId());
    }
  }, [updateVitalsModal.isOpen]);

  if (!patient) return null;

  const vitals: VitalSigns = {
    heart_rate: optionalNumber(hr, true),
    systolic_bp: optionalNumber(bpSys, true),
    diastolic_bp: optionalNumber(bpDia, true),
    respiratory_rate: optionalNumber(rr, true),
    oxygen_saturation: optionalNumber(spo2),
    temperature_celsius: optionalNumber(temp),
    pain_score: optionalNumber(pain, true),
  };
  const nothingEntered = Object.values(vitals).every((v) => v == null);
  const close = () => setUpdateVitalsModal({ isOpen: false, patientId: null });

  const handleRunReTriage = async () => {
    const numHr = vitals.heart_rate;
    if (numHr != null && (numHr > 220 || numHr < 30)) {
      if (!window.confirm(`WARNING: Heart rate of ${numHr} is extremely abnormal. Are you sure this is not a typo?`)) return;
    }
    setIsLoading(true);
    setError(null);
    setOfflineNote(null);
    try {
      const view = fromTriageResponse(await reassessVisit(patient.id, { request_id: requestId, vitals }));
      if (view.requiresConfirmation) {
        setPending(view); // asymmetric cost: a downgrade always needs explicit confirmation
      } else {
        await refresh();
        close();
      }
    } catch (err) {
      if (!shouldUseEdgeFallback(err)) {
        setError(describeError(err)); // 4xx: show it, never mask it with the offline model
      } else {
        // Backend unreachable. Show offline guidance, but NOTHING is saved: the server is the source of truth.
        try {
          const edge = await runEdgeInference({ age_years: patient.age, ...vitals, complaint: patient.complaint });
          setOfflineNote(`Server unreachable: these vitals were NOT saved. Offline model suggests ESI ${edge.recommended_esi}. ${edge.rationale}`);
        } catch {
          setOfflineNote("Server unreachable: these vitals were NOT saved and no offline guidance is available. Assess manually.");
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  const decide = async (esi: number, reason: string) => {
    try {
      await overrideEsi({ visit_id: patient.id, assigned_esi: esi, override_reason: reason });
      await refresh();
      close();
    } catch (err) {
      setError(describeError(err));
    }
  };

  const field = (label: string, value: string, set: (v: string) => void, placeholder: string, step?: string) => (
    <div>
      <label className="text-xs font-bold text-slate-500">{label}</label>
      <input type="number" step={step} value={value} onChange={(e) => set(e.target.value)} className="w-full p-2 border rounded" placeholder={placeholder} />
    </div>
  );

  return (
    <Dialog open={updateVitalsModal.isOpen} onOpenChange={(open) => setUpdateVitalsModal({ isOpen: open, patientId: null })}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Update Vitals: {patient.name}</DialogTitle>
        </DialogHeader>
        {error && <p className="rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{error}</p>}
        {offlineNote && <p className="rounded border border-amber-300 bg-amber-50 p-2 text-sm text-amber-800">{offlineNote}</p>}
        {!pending ? (
          <div className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-4">
              {field("Heart Rate", hr, setHr, "BPM")}
              {field("SpO2 %", spo2, setSpo2, "%")}
              {field("BP Systolic", bpSys, setBpSys, "mmHg")}
              {field("BP Diastolic", bpDia, setBpDia, "mmHg")}
              {field("Respiratory Rate", rr, setRr, "breaths/min")}
              {field("Temperature (°C)", temp, setTemp, "°C", "0.1")}
              {field("Pain (0-10)", pain, setPain, "0-10")}
            </div>
            <p className="text-xs text-muted-foreground">Leave a field blank if it was not re-measured; the previous reading is kept.</p>
            <button onClick={handleRunReTriage} disabled={isLoading || nothingEntered} className="w-full py-2 bg-blue-600 text-white font-bold rounded hover:bg-blue-700 disabled:opacity-50">
              {isLoading ? "Re-running assessment..." : offlineNote ? "Retry" : "Submit & Re-Triage"}
            </button>
          </div>
        ) : (
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg space-y-4">
            <div className="flex items-start gap-2">
              <span className="text-2xl">⚠️</span>
              <div>
                <h4 className="font-bold text-yellow-800">Down-Triage Suggested</h4>
                <p className="text-sm text-yellow-700">
                  The assessment now suggests <strong>ESI {pending.recommended_esi}</strong> instead of <strong>ESI {patient.esi}</strong>. The new vitals are saved; the level stays at ESI {patient.esi} until you decide.
                </p>
                <p className="text-xs text-yellow-700 mt-2">{pending.rationale}</p>
                <p className="text-xs text-yellow-600 mt-2 italic">Hospital policy requires explicit clinical confirmation to lower an assigned ESI level.</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => void decide(pending.recommended_esi, `Clinician confirmed AI down-triage from ESI ${patient.esi} after vitals update`)} className="flex-1 py-2 bg-yellow-600 text-white font-bold rounded hover:bg-yellow-700">
                Confirm Downgrade
              </button>
              <button onClick={() => { void refresh(); close(); }} className="flex-1 py-2 bg-slate-200 text-slate-800 font-bold rounded hover:bg-slate-300">
                Keep ESI {patient.esi}
              </button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
