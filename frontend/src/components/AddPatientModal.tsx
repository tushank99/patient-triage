/* eslint-disable prettier/prettier */
import React, { useState, useEffect, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { useTriage, EsiLevel } from "../lib/triage-context";
import { useVoiceToText } from "../hooks/useVoiceToText";
import { runEdgeInference } from "../lib/onnx-engine";
import { PatientDocumentsModal } from "./PatientDocumentsModal";
import { TranscriptRecorder } from "./TranscriptRecorder";
import {
  describeError,
  lookupPatient,
  newRequestId,
  optionalNumber,
  overrideEsi,
  shouldUseEdgeFallback,
  triagePatient,
  type PatientLookup,
  type VitalSigns,
} from "../lib/api";
import { fromTriageResponse, offlineDefault, pct, type AssessmentView } from "../lib/assessment-view";

export function AddPatientModal() {
  const { isAddModalOpen, setIsAddModalOpen, refresh } = useTriage();

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [isLoading, setIsLoading] = useState(false);

  // Voice-to-Text Hook
  const {
    isListening,
    transcript,
    startListening,
    stopListening,
    error: voiceError,
    setTranscript
  } = useVoiceToText();

  // Form State
  const [name, setName] = useState("");
  const [gender, setGender] = useState<"M" | "F" | "O" | "">("");
  const [dob, setDob] = useState("");
  const [complaint, setComplaint] = useState("");

  const [temp, setTemp] = useState("");
  const [tempUnit, setTempUnit] = useState<"C" | "F">("C");
  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [hr, setHr] = useState("");
  const [spo2, setSpo2] = useState("");
  const [pain, setPain] = useState("");
  const [rr, setRr] = useState("");

  // Returning patient: the MRN is looked up on the server; "has history" is computed there.
  const [mrnInput, setMrnInput] = useState("");
  const [lookup, setLookup] = useState<PatientLookup | null>(null);
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [isLookingUp, setIsLookingUp] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [transcriptCaptureId, setTranscriptCaptureId] = useState<string | null>(null);

  // One request id per intake; retries reuse it, so a double click cannot create two visits.
  const [requestId, setRequestId] = useState(newRequestId);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [aiResult, setAiResult] = useState<AssessmentView | null>(null);

  // Override Tracking State
  const [isOverriding, setIsOverriding] = useState(false);
  const [selectedOverrideEsi, setSelectedOverrideEsi] = useState<EsiLevel | null>(null);
  const [overrideReason, setOverrideReason] = useState("");

  const OVERRIDE_REASONS = [
    "Patient appearance toxic/lethargic",
    "Clinical intuition / Gestalt",
    "Unreported complex medical history",
    "AI under-triaged (Too low)",
    "AI over-triaged (Too high)"
  ];

  // Append voice transcript to complaint automatically
  useEffect(() => {
    if (transcript) {
      setComplaint((prev) => (prev ? prev + " " + transcript : transcript));
      setTranscript(""); // clear the buffer so it doesn't duplicate
    }
  }, [transcript, setTranscript]);

  const resetForm = () => {
    setStep(1);
    setName("");
    setGender("");
    setDob("");
    setComplaint("");
    setTemp("");
    setTempUnit("C");
    setBpSys("");
    setBpDia("");
    setHr("");
    setSpo2("");
    setPain("");
    setRr("");
    setMrnInput("");
    setLookup(null);
    setLookupError(null);
    setRequestId(newRequestId());
    setSubmitError(null);
    setTranscriptCaptureId(null);

    // AI and Override Resets
    setAiResult(null);
    setIsOverriding(false);
    setSelectedOverrideEsi(null);
    setOverrideReason("");
    stopListening();
  };

  const handleOpenChange = (open: boolean) => {
    setIsAddModalOpen(open);
    if (!open) {
      resetForm();
    }
  };

  // Derived calculations
  const age = useMemo(() => {
    if (!dob) return null;
    const birthDate = new Date(dob);
    const today = new Date();
    let calculatedAge = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      calculatedAge--;
    }
    if (calculatedAge === 0) {
      const diffTime = Math.abs(today.getTime() - birthDate.getTime());
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      return Number((diffDays / 365.25).toFixed(2));
    }
    return calculatedAge;
  }, [dob]);

  const numTemp = temp ? parseFloat(temp) : null;
  const numBpSys = bpSys ? parseInt(bpSys, 10) : null;
  const numBpDia = bpDia ? parseInt(bpDia, 10) : null;
  const numHr = hr ? parseInt(hr, 10) : null;
  const numSpo2 = optionalNumber(spo2);
  const numRr = optionalNumber(rr, true);

  let errTemp = false;
  let normalizedTemp = numTemp;
  if (numTemp !== null) {
    if (tempUnit === "C") {
      errTemp = numTemp < 30 || numTemp > 45;
    } else {
      errTemp = numTemp < 86 || numTemp > 113;
      normalizedTemp = (numTemp - 32) * 5 / 9;
    }
  }

  const errHr = numHr !== null && (numHr < 20 || numHr > 300);
  const errSpo2 = numSpo2 !== null && (numSpo2 < 50 || numSpo2 > 100);
  const errBpSys = numBpSys !== null && (numBpSys < 40 || numBpSys > 300);
  const errBpDia = numBpDia !== null && (numBpDia < 20 || numBpDia > 200);
  const errBpLogic = numBpSys !== null && numBpDia !== null && numBpDia >= numBpSys;
  const errRr = numRr !== null && (numRr < 4 || numRr > 80);

  const hasErrors = errTemp || errHr || errSpo2 || errBpSys || errBpDia || errBpLogic || errRr;
  const hasMandatory = name.trim() !== "" && gender !== "" && dob !== "" && complaint.trim() !== "";
  const canProceed = hasMandatory && !hasErrors;

  const skippedVitals = [];
  if (!temp) skippedVitals.push("Temperature");
  if (!bpSys || !bpDia) skippedVitals.push("Blood Pressure");
  if (!hr) skippedVitals.push("Heart Rate");
  if (!spo2) skippedVitals.push("Oxygen Saturation");
  if (!rr) skippedVitals.push("Respiratory Rate");
  if (!pain) skippedVitals.push("Pain Score");

  const handleLookup = async () => {
    if (!mrnInput.trim()) return;
    setIsLookingUp(true);
    setLookupError(null);
    try {
      const found = await lookupPatient(mrnInput);
      setLookup(found);
      // Pre-fill demographics from the record; the nurse can still correct them.
      if (found.name) setName(found.name);
      setGender(found.gender);
      if (found.date_of_birth) setDob(found.date_of_birth);
    } catch (err) {
      setLookup(null);
      setLookupError(describeError(err));
    } finally {
      setIsLookingUp(false);
    }
  };

  const clearLookup = () => {
    setLookup(null);
    setMrnInput("");
    setLookupError(null);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setSubmitError(null);

    const vitals: VitalSigns = {
      heart_rate: numHr,
      systolic_bp: numBpSys,
      diastolic_bp: numBpDia,
      respiratory_rate: numRr,
      oxygen_saturation: numSpo2,
      temperature_celsius: normalizedTemp !== null ? Math.round(normalizedTemp * 10) / 10 : null,
      pain_score: optionalNumber(pain, true),
    };
    // Keep fractional age: a 3-month-old is 0.25, not 0 (pediatric rules depend on it).
    const ageYears = age ?? 0;

    try {
      const response = await triagePatient({
        request_id: requestId,
        mrn: lookup?.mrn ?? null,
        patient_name: name,
        age_years: ageYears,
        gender: gender || "O",
        chief_complaint: complaint,
        vitals,
        transcript_capture_id: transcriptCaptureId,
      });
      setAiResult(fromTriageResponse(response));
      setStep(3);
    } catch (err) {
      if (!shouldUseEdgeFallback(err)) {
        // 4xx: wrong input or a contract bug. Show it; never hide it behind the offline model.
        setSubmitError(describeError(err));
        return;
      }
      console.error("Backend unavailable, using the local ONNX model:", err);
      try {
        setAiResult(await runEdgeInference({ age_years: ageYears, ...vitals, complaint }));
      } catch (onnxErr) {
        console.error("ONNX fallback also failed:", onnxErr);
        setAiResult(offlineDefault());
      }
      setStep(3);
    } finally {
      setIsLoading(false);
    }
  };

  /** The visit already exists on the server (created by /api/triage). Finalising either accepts the AI level or
   *  records a human override, then refreshes the queue from the server. */
  const handleFinalizeTriage = async (finalEsi: EsiLevel, overrideReasonText?: string) => {
    if (!aiResult?.visitId) return;
    if (overrideReasonText && finalEsi !== aiResult.finalEsi) {
      try {
        await overrideEsi({ visit_id: aiResult.visitId, assigned_esi: finalEsi, override_reason: overrideReasonText });
      } catch (err) {
        setSubmitError(`Override was not saved: ${describeError(err)}`);
        return;
      }
    }
    await refresh();
    setIsAddModalOpen(false);
    resetForm();
  };

  return (
    <Dialog open={isAddModalOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-2xl bg-white p-0 overflow-hidden sm:rounded-xl">
        {step === 1 ? (
          <div className="flex flex-col h-full max-h-[85vh]">
            <DialogHeader className="px-6 py-5 border-b border-border bg-white">
              <DialogTitle className="text-xl">Add New Patient</DialogTitle>
            </DialogHeader>
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {/* Section A: Mandatory Fields */}
              <section className="space-y-4">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Mandatory Information</h3>

                <div className="space-y-4">
                  <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                    <label htmlFor="patient-mrn" className="block text-sm font-medium text-foreground mb-1">Returning patient? Look up MRN</label>
                    {lookup ? (
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span>
                          <span className="font-semibold">{lookup.mrn}</span> · {lookup.name ?? "Unnamed"} ·{" "}
                          {lookup.has_history ? `${lookup.document_count} history documents on file` : "no history documents on file"}
                        </span>
                        <span className="flex gap-3">
                          <button type="button" onClick={() => setDocsOpen(true)} className="text-xs font-semibold text-blue-600 hover:underline">Documents</button>
                          <button type="button" onClick={clearLookup} className="text-xs font-medium text-muted-foreground hover:text-foreground">Clear</button>
                        </span>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <input
                          id="patient-mrn"
                          type="text"
                          value={mrnInput}
                          onChange={(e) => setMrnInput(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void handleLookup(); } }}
                          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500"
                          placeholder="e.g. MRN-GER-005 (leave blank for a new walk-in)"
                        />
                        <button type="button" onClick={() => void handleLookup()} disabled={!mrnInput.trim() || isLookingUp} className="rounded-md bg-slate-800 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50">
                          {isLookingUp ? "Looking up..." : "Look up"}
                        </button>
                      </div>
                    )}
                    {lookupError && <p className="mt-1 text-xs font-medium text-red-600">{lookupError}</p>}
                  </div>

                  <div>
                    <label htmlFor="patient-name" className="block text-sm font-medium text-foreground mb-1">Full Name</label>
                    <input
                      id="patient-name"
                      autoFocus
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-shadow"
                      placeholder="e.g. Smith, John"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="patient-gender" className="block text-sm font-medium text-foreground mb-1">Gender</label>
                      <select
                        id="patient-gender"
                        value={gender}
                        onChange={(e) => setGender(e.target.value as "M" | "F" | "O" | "")}
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-shadow"
                      >
                        <option value="" disabled>Select gender</option>
                        <option value="M">Male</option>
                        <option value="F">Female</option>
                        <option value="O">Other</option>
                      </select>
                    </div>
                    <div>
                      <label htmlFor="patient-dob" className="block text-sm font-medium text-foreground mb-1">Date of Birth</label>
                      <div className="flex items-center gap-3">
                        <input
                          id="patient-dob"
                          type="date"
                          value={dob}
                          onChange={(e) => setDob(e.target.value)}
                          max={new Date().toISOString().split("T")[0]}
                          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-shadow"
                        />
                        {age !== null && (
                          <span className="text-sm font-medium text-muted-foreground tabular-nums shrink-0">
                            {age < 1 ? `${Math.floor(age * 12)} mo` : `${Math.floor(age)} yr`}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Voice Enabled Chief Complaint Box */}
                  <div className="relative">
                    <label htmlFor="patient-complaint" className="block text-sm font-medium text-foreground mb-1">Chief Complaint</label>
                    <textarea
                      id="patient-complaint"
                      rows={3}
                      value={complaint}
                      onChange={(e) => setComplaint(e.target.value)}
                      placeholder="Describe presenting symptoms and clinical observations..."
                      className="w-full rounded-md border border-input bg-background px-3 py-2 pb-10 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-shadow resize-none"
                    />
                    <button
                      type="button"
                      onClick={isListening ? stopListening : startListening}
                      className={`absolute bottom-3 right-3 px-3 py-1 text-xs font-medium rounded-full shadow-sm transition-colors ${
                        isListening
                          ? 'bg-red-500 text-white animate-pulse'
                          : 'bg-slate-200 text-slate-700 hover:bg-slate-300'
                      }`}
                      title="Dictate Symptoms"
                    >
                      {isListening ? "⏹ Stop Dictating" : "🎤 Dictate"}
                    </button>
                  </div>
                  {voiceError && <span className="text-xs text-red-500 block mt-1">{voiceError}</span>}
                  <TranscriptRecorder
                    mrn={lookup?.mrn}
                    onConfirmed={(text, id) => { setComplaint((prev) => (prev ? `${prev} ${text}` : text)); setTranscriptCaptureId(id); }}
                  />
                </div>
              </section>

              {/* Section B: Optional Vitals */}
              <section className="rounded-lg border border-slate-200 bg-slate-50 p-5 space-y-4">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Optional Vitals</h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4">
                  {/* Temp */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label htmlFor="vital-temp" className="block text-sm font-medium text-foreground">Temperature</label>
                      <div className="flex items-center rounded-md border border-input bg-background p-0.5" role="group">
                        <button type="button" onClick={() => setTempUnit("C")} className={`px-2 py-0.5 text-[10px] font-bold rounded-sm transition-colors ${tempUnit === "C" ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>°C</button>
                        <button type="button" onClick={() => setTempUnit("F")} className={`px-2 py-0.5 text-[10px] font-bold rounded-sm transition-colors ${tempUnit === "F" ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>°F</button>
                      </div>
                    </div>
                    <input
                      id="vital-temp"
                      type="number"
                      step="0.1"
                      value={temp}
                      onChange={(e) => setTemp(e.target.value)}
                      className={`w-full rounded-md border bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-shadow ${errTemp ? "border-red-500 focus:border-red-500" : "border-input focus:border-blue-500"}`}
                      placeholder={tempUnit === "C" ? "e.g. 37.2" : "e.g. 98.6"}
                    />
                    {errTemp && <p className="mt-1 text-xs text-red-500 font-medium">Out of normal range ({tempUnit === "C" ? "30-45" : "86-113"})</p>}
                  </div>

                  {/* BP */}
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">Blood Pressure (mmHg)</label>
                    <div className="flex items-center gap-2">
                      <div className="flex-1">
                        <input
                          type="number"
                          value={bpSys}
                          onChange={(e) => setBpSys(e.target.value)}
                          className={`w-full rounded-md border bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-shadow ${errBpSys || errBpLogic ? "border-red-500 focus:border-red-500" : "border-input focus:border-blue-500"}`}
                          placeholder="Sys"
                        />
                      </div>
                      <span className="text-muted-foreground font-medium">/</span>
                      <div className="flex-1">
                        <input
                          type="number"
                          value={bpDia}
                          onChange={(e) => setBpDia(e.target.value)}
                          className={`w-full rounded-md border bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-shadow ${errBpDia || errBpLogic ? "border-red-500 focus:border-red-500" : "border-input focus:border-blue-500"}`}
                          placeholder="Dia"
                        />
                      </div>
                    </div>
                    {errBpSys && <p className="mt-1 text-xs text-red-500 font-medium">Sys out of range (40-300)</p>}
                    {!errBpSys && errBpDia && <p className="mt-1 text-xs text-red-500 font-medium">Dia out of range (20-200)</p>}
                    {!errBpSys && !errBpDia && errBpLogic && <p className="mt-1 text-xs text-red-500 font-medium">Diastolic cannot be &ge; Systolic</p>}
                  </div>

                  {/* HR */}
                  <div>
                    <label htmlFor="vital-hr" className="block text-sm font-medium text-foreground mb-1">Heart Rate (BPM)</label>
                    <input
                      id="vital-hr"
                      type="number"
                      value={hr}
                      onChange={(e) => setHr(e.target.value)}
                      className={`w-full rounded-md border bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-shadow ${errHr ? "border-red-500 focus:border-red-500" : "border-input focus:border-blue-500"}`}
                      placeholder="e.g. 80"
                    />
                    {errHr && <p className="mt-1 text-xs text-red-500 font-medium">Out of normal human range (20-300)</p>}
                  </div>

                  {/* SpO2 */}
                  <div>
                    <label htmlFor="vital-spo2" className="block text-sm font-medium text-foreground mb-1">Oxygen Saturation (SpO2 %)</label>
                    <input
                      id="vital-spo2"
                      type="number"
                      value={spo2}
                      onChange={(e) => setSpo2(e.target.value)}
                      className={`w-full rounded-md border bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-shadow ${errSpo2 ? "border-red-500 focus:border-red-500" : "border-input focus:border-blue-500"}`}
                      placeholder="e.g. 98"
                    />
                    {errSpo2 && <p className="mt-1 text-xs text-red-500 font-medium">Out of normal human range (50-100)</p>}
                  </div>

                  {/* Pain */}
                  <div>
                    <label htmlFor="vital-pain" className="block text-sm font-medium text-foreground mb-1">Pain Score (1-10)</label>
                    <select
                      id="vital-pain"
                      value={pain}
                      onChange={(e) => setPain(e.target.value)}
                      className="w-full rounded-md border border-input bg-white px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-shadow"
                    >
                      <option value="">None</option>
                      {[1,2,3,4,5,6,7,8,9,10].map(n => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </div>

                  {/* RR */}
                  <div>
                    <label htmlFor="vital-rr" className="block text-sm font-medium text-foreground mb-1">Respiratory Rate (breaths/min)</label>
                    <input
                      id="vital-rr"
                      type="number"
                      value={rr}
                      onChange={(e) => setRr(e.target.value)}
                      className={`w-full rounded-md border bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-shadow ${errRr ? "border-red-500 focus:border-red-500" : "border-input focus:border-blue-500"}`}
                      placeholder="e.g. 16"
                    />
                    {errRr && <p className="mt-1 text-xs text-red-500 font-medium">Out of range (4-80)</p>}
                  </div>
                </div>

                <p className="pt-2 text-xs font-medium text-muted-foreground italic">
                  *Note: Skipping vitals will result in a Low Confidence AI Score.
                </p>
              </section>
            </div>
            <div className="p-6 border-t border-border bg-white flex justify-end">
              <button
                type="button"
                onClick={() => setStep(2)}
                disabled={!canProceed}
                className="rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Review Patient Details
              </button>
            </div>
          </div>
        ) : step === 2 ? (
          <div className="flex flex-col h-full max-h-[85vh]">
            <DialogHeader className="px-6 py-5 border-b border-border bg-white">
              <DialogTitle className="text-xl">Review & Confirm</DialogTitle>
            </DialogHeader>
            <div className="flex-1 overflow-y-auto p-6 space-y-6">

              {skippedVitals.length > 0 && (
                <div className="rounded-md bg-amber-50 border border-amber-200 p-4">
                  <div className="flex items-start gap-3">
                    <span className="text-amber-600 text-lg leading-none">⚠️</span>
                    <div>
                      <h4 className="text-sm font-bold text-amber-800">Incomplete Data Warning</h4>
                      <p className="text-sm text-amber-700 mt-1">
                        You skipped: <span className="font-semibold">{skippedVitals.join(", ")}</span>. This will reduce AI scoring confidence.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div className="rounded-xl border border-border bg-card shadow-sm overflow-hidden">
                <div className="p-5 border-b border-border bg-slate-50/50">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">Patient Demographics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                      <p className="text-xs text-muted-foreground">Full Name</p>
                      <p className="font-semibold text-foreground">{name}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Gender</p>
                      <p className="font-semibold text-foreground">{gender === "M" ? "Male" : gender === "F" ? "Female" : "Other"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Calculated Age</p>
                      <p className="font-semibold text-foreground">
                        {age !== null ? (age < 1 ? `${Math.floor(age * 12)} months` : `${Math.floor(age)} years`) : "Unknown"}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-5 border-b border-border">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">Chief Complaint</h3>
                  <blockquote className="border-l-4 border-blue-500 bg-blue-50/30 pl-4 py-2 italic text-sm text-foreground">
                    "{complaint}"
                  </blockquote>
                </div>

                <div className="p-5">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">Vitals</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-y-4 gap-x-2">
                    <div>
                      <p className="text-xs text-muted-foreground">Temperature</p>
                      <p className="font-semibold text-foreground">{temp ? `${temp} °${tempUnit}` : "—"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Blood Pressure</p>
                      <p className="font-semibold text-foreground">{bpSys && bpDia ? `${bpSys}/${bpDia} mmHg` : "—"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Heart Rate</p>
                      <p className="font-semibold text-foreground">{hr ? `${hr} bpm` : "—"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">SpO2</p>
                      <p className="font-semibold text-foreground">{spo2 ? `${spo2} %` : "—"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Pain Score</p>
                      <p className="font-semibold text-foreground">{pain ? `${pain}/10` : "—"}</p>
                    </div>
                    <div className="col-span-2 md:col-span-3">
                      <p className="text-xs text-muted-foreground">Respiratory Rate</p>
                      <p className="font-semibold text-foreground">{rr ? `${rr} /min` : "—"}</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-xs text-muted-foreground">Record</p>
                      <p className="font-semibold text-foreground">
                        {lookup ? `${lookup.mrn} · ${lookup.has_history ? `${lookup.document_count} history documents` : "no history on file"}` : "New walk-in (MRN assigned on submit)"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

            </div>
            {submitError && (
              <p className="mx-6 mb-2 rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{submitError}</p>
            )}
            <div className="p-6 border-t border-border bg-white flex justify-between items-center">
              <button
                type="button"
                onClick={() => setStep(1)}
                disabled={isLoading}
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
              >
                &larr; Edit Details
              </button>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isLoading}
                className="rounded-md bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:opacity-70 disabled:cursor-wait"
              >
                {isLoading ? "Running AI Models..." : "Confirm & Run AI Triage"}
              </button>
            </div>
          </div>
        ) : step === 3 && aiResult ? (
          <div className="flex flex-col h-full max-h-[85vh]">
            <DialogHeader className="px-6 py-5 border-b border-border bg-white">
              <DialogTitle className="text-xl">AI Assessment Result</DialogTitle>
            </DialogHeader>
            <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50">

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* ESI Recommendation Card */}
                <div className="md:col-span-1 flex flex-col items-center justify-center p-6 bg-white border border-border rounded-xl shadow-sm">
                  <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-2">Recommended</p>
                  <div className={`flex items-center justify-center w-24 h-24 rounded-full text-4xl font-black shadow-inner
                    ${aiResult.recommended_esi === 1 ? 'bg-red-100 text-red-700' :
                      aiResult.recommended_esi === 2 ? 'bg-orange-100 text-orange-700' :
                      aiResult.recommended_esi === 3 ? 'bg-yellow-100 text-yellow-700' :
                      aiResult.recommended_esi === 4 ? 'bg-green-100 text-green-700' :
                      'bg-blue-100 text-blue-700'}`}
                  >
                    {aiResult.recommended_esi}
                  </div>
                  <p className="mt-3 text-sm font-semibold text-foreground">ESI Level</p>
                </div>

                {/* Confidence & Badges Card */}
                <div className="md:col-span-2 space-y-4">
                  <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                    <h3 className="text-sm font-bold uppercase text-muted-foreground mb-3">Confidence (three separate signals)</h3>
                    {([
                      ["Model probability", aiResult.confidence.model_probability, "Top-class probability of the ML model (not calibrated)."],
                      ["Data completeness", aiResult.confidence.data_completeness, aiResult.confidence.missing_critical.length ? `Missing critical: ${aiResult.confidence.missing_critical.join(", ")}` : "Share of vitals actually measured."],
                      ["History retrieved", aiResult.confidence.retrieval_confidence, aiResult.confidence.retrieval_note],
                    ] as [string, number | null | undefined, string][]).map(([label, value, note]) => (
                      <div key={label} className="mb-2.5">
                        <div className="flex justify-between text-xs font-semibold">
                          <span>{label}</span><span className="tabular-nums">{pct(value)}</span>
                        </div>
                        <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                          <div className={`h-2 rounded-full ${value == null ? "bg-slate-300" : value >= 0.8 ? "bg-green-500" : value >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`} style={{ width: `${Math.round((value ?? 0) * 100)}%` }} />
                        </div>
                        <p className="mt-0.5 text-[11px] text-muted-foreground">{note}</p>
                      </div>
                    ))}

                    {/* Deterministic Badges */}
                    {aiResult.badges && aiResult.badges.length > 0 && (
                      <div className="mt-5">
                        <h4 className="text-xs font-bold uppercase text-muted-foreground mb-2">Clinical Flags</h4>
                        <div className="flex flex-wrap gap-2">
                          {aiResult.badges.map((badge, i) => (
                            <span key={i} className={`px-2.5 py-1 text-xs font-semibold rounded-full border
                              ${badge.type === 'danger' ? 'bg-red-50 text-red-700 border-red-200' :
                                badge.type === 'warning' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                                'bg-emerald-50 text-emerald-700 border-emerald-200'}`}
                            >
                              {badge.type === 'danger' ? '🔴 ' : badge.type === 'warning' ? '⚠️ ' : '🟢 '}{badge.label}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Plain-English Rationale */}
              <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">AI Rationale</h3>
                <p className="text-sm text-foreground leading-relaxed font-medium">
                  {aiResult.rationale}
                </p>
              </div>

              {aiResult.reviewReasons.length > 0 && (
                <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 space-y-1">
                  <p className="font-bold">Clinician review recommended</p>
                  {aiResult.reviewReasons.map((r) => <p key={r}>• {r}</p>)}
                </div>
              )}

              {aiResult.riskFactors.length > 0 && (
                <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-2">Risk Factors</h3>
                  <ul className="list-disc pl-5 text-sm space-y-0.5">{aiResult.riskFactors.map((r) => <li key={r}>{r}</li>)}</ul>
                </div>
              )}

              {aiResult.facts && (aiResult.facts.allergies.length > 0 || aiResult.facts.medications.length > 0 || aiResult.facts.conflicts.length > 0) && (
                <div className="p-5 bg-white border border-border rounded-xl shadow-sm space-y-2 text-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">From the record (extracted, not AI)</h3>
                  {aiResult.facts.conflicts.map((c) => (
                    <p key={c.description} className={`rounded border p-2 text-xs ${c.severity === "high" ? "border-red-200 bg-red-50 text-red-800" : "border-amber-200 bg-amber-50 text-amber-800"}`}>
                      <strong>Conflict:</strong> {c.description}
                    </p>
                  ))}
                  <p><span className="font-semibold">Allergies:</span> {aiResult.facts.allergies.length ? aiResult.facts.allergies.map((a) => `${a.substance} (${a.reaction})`).join("; ") : (aiResult.facts.nkda_statements.length ? "NKDA" : "not documented")}</p>
                  <p><span className="font-semibold">Current medications:</span> {aiResult.facts.medications.filter((m) => m.effective_status === "active").map((m) => `${m.name} ${m.dose}`.trim()).join("; ") || "none documented"}</p>
                  {aiResult.facts.medications.some((m) => m.effective_status === "discontinued") && (
                    <p className="text-xs text-muted-foreground">Stopped: {aiResult.facts.medications.filter((m) => m.effective_status === "discontinued").map((m) => m.name).join(", ")}</p>
                  )}
                </div>
              )}

              {aiResult.claims && aiResult.claims.length > 0 && (
                <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-2">AI statements checked against the record</h3>
                  <ul className="space-y-1 text-sm">
                    {aiResult.claims.map((c, i) => (
                      <li key={i} className="flex gap-2">
                        <span title={c.reasons.join("; ")} className={`shrink-0 rounded px-1.5 text-[11px] font-bold ${c.verdict === "supported" ? "bg-emerald-100 text-emerald-800" : c.verdict === "partial" ? "bg-amber-100 text-amber-800" : "bg-red-100 text-red-800"}`}>
                          {c.verdict === "supported" ? "✓ cited" : c.verdict === "partial" ? "~ partial" : "✗ " + c.verdict.replace("_", " ")}
                        </span>
                        <span className={c.verdict === "supported" || c.verdict === "partial" ? "" : "line-through decoration-red-400"}>{c.text}</span>
                        <span className="text-xs text-muted-foreground">{c.citations.join(", ")}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {aiResult.mlDrivers && aiResult.mlDrivers.length > 0 && (
                <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-2">What drove the ML model</h3>
                  <ul className="space-y-1 text-sm">
                    {aiResult.mlDrivers.map((d) => (
                      <li key={d.feature} className="flex justify-between gap-4">
                        <span>{d.label}{d.value != null && !d.feature.endsWith("_missing") ? ` = ${d.value}` : ""}</span>
                        <span className={`font-mono text-xs ${d.contribution >= 0 ? "text-emerald-700" : "text-red-700"}`}>{d.contribution >= 0 ? "+" : ""}{d.contribution.toFixed(2)}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[11px] text-muted-foreground">TreeSHAP, in log-odds for the model's predicted ESI (positive = pushed toward it). Explains the model, not the patient.</p>
                </div>
              )}

              {(aiResult.warnings.length > 0 || aiResult.degraded.length > 0) && (
                <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 space-y-1">
                  {aiResult.warnings.map((w) => <p key={w}>⚠️ {w}</p>)}
                  {aiResult.degraded.length > 0 && <p>Unavailable this time: {aiResult.degraded.join(", ")}</p>}
                </div>
              )}

              {aiResult.rulesFired.length > 0 && (
                <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">Safety Rules Fired</h3>
                  <ul className="space-y-1 text-sm">
                    {aiResult.rulesFired.map((r) => (
                      <li key={r.id}><span className="font-semibold">ESI ≤ {r.esi}</span> · {r.reason} <span className="text-xs text-muted-foreground">({r.id} v{r.version})</span></li>
                    ))}
                  </ul>
                </div>
              )}

              {/* What the AI was shown, so the nurse can check it */}
              {aiResult.source === "cloud" && (
                <div className="p-5 bg-white border border-border rounded-xl shadow-sm">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-3">
                    History Shown to the AI {aiResult.mrn && <span className="normal-case font-medium">({aiResult.mrn})</span>}
                  </h3>
                  {aiResult.historyLowConfidence && (
                    <p className="mb-2 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
                      No note in this patient's history matched the complaint confidently, so only allergies, medications and the problem list were used.
                    </p>
                  )}
                  {aiResult.hasHistory ? (
                    <ul className="space-y-2">
                      {aiResult.history.map((doc) => (
                        <li key={doc.id}>
                          <details className="text-sm">
                            <summary className="cursor-pointer font-medium">{doc.title} <span className="text-xs text-muted-foreground">{doc.doc_date ?? ""}{doc.section ? ` · ${doc.section}` : ""}{doc.page ? ` · p.${doc.page}` : ""}</span></summary>
                            <p className="mt-1 whitespace-pre-line text-xs text-muted-foreground">{doc.snippet}</p>
                          </details>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-muted-foreground">No prior history on file. Assessed on vitals and complaint only.</p>
                  )}
                </div>
              )}

              {aiResult.modelNote && <p className="text-[11px] text-muted-foreground">ML model: {aiResult.modelNote}{aiResult.modelNote.includes("synthetic") ? " (demo only: not trained on real patients)" : ""}</p>}

              {/* Override Workflow Inline */}
              {isOverriding && (
                <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl shadow-lg animate-in slide-in-from-top-2">
                  {!selectedOverrideEsi ? (
                    <>
                      <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                        <span>⚠️</span> Select New ESI Level:
                      </h4>
                      <div className="flex gap-2">
                        {[1, 2, 3, 4, 5].map((level) => (
                          <button
                            key={level}
                            onClick={() => setSelectedOverrideEsi(level as EsiLevel)}
                            className={`flex-1 py-3 rounded-md font-bold transition-all shadow-sm
                              ${level === aiResult.recommended_esi
                                ? 'bg-slate-700 text-slate-400 cursor-not-allowed border-none'
                                : 'bg-slate-800 text-white border border-slate-700 hover:bg-slate-700 hover:border-slate-500'}`}
                            disabled={level === aiResult.recommended_esi}
                          >
                            ESI {level}
                          </button>
                        ))}
                      </div>
                      <button onClick={() => setIsOverriding(false)} className="mt-4 text-xs text-slate-400 hover:text-white">
                        Cancel Override
                      </button>
                    </>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <h4 className="text-sm font-bold text-white">
                          Reason for overriding AI to ESI {selectedOverrideEsi}?
                        </h4>
                        <button onClick={() => setSelectedOverrideEsi(null)} className="text-xs text-slate-400 hover:text-white">
                          &larr; Back
                        </button>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {OVERRIDE_REASONS.map((reason) => (
                          <button
                            key={reason}
                            onClick={() => setOverrideReason(reason)}
                            className={`px-3 py-2 text-xs text-left rounded border transition-colors ${
                              overrideReason === reason
                                ? 'bg-blue-600 border-blue-500 text-white'
                                : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            {reason}
                          </button>
                        ))}
                      </div>

                      <button
                        onClick={() => void handleFinalizeTriage(selectedOverrideEsi, overrideReason)}
                        disabled={!overrideReason}
                        className="w-full mt-2 rounded-md bg-red-600 px-4 py-2.5 text-sm font-bold text-white shadow-sm transition-colors hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Confirm Override & Submit
                      </button>
                    </div>
                  )}
                </div>
              )}

            </div>
            {submitError && <p className="mx-6 mb-2 rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{submitError}</p>}
            <div className="p-6 border-t border-border bg-white flex justify-between items-center">
              <button
                type="button"
                onClick={() => setIsOverriding(true)}
                disabled={isOverriding || !aiResult.visitId}
                className="text-sm font-semibold text-slate-600 hover:text-red-600 transition-colors disabled:opacity-50"
              >
                Manual Override
              </button>
              {aiResult.visitId ? (
                <button
                  type="button"
                  onClick={() => void handleFinalizeTriage(aiResult.recommended_esi as EsiLevel)}
                  disabled={isOverriding}
                  className="rounded-md bg-green-600 px-6 py-2.5 text-sm font-bold text-white shadow-sm transition-colors hover:bg-green-700 disabled:opacity-50"
                >
                  Accept AI Decision
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => { setAiResult(null); setStep(2); }}
                  className="rounded-md bg-amber-600 px-6 py-2.5 text-sm font-bold text-white shadow-sm hover:bg-amber-700"
                >
                  Not saved: retry connection
                </button>
              )}
            </div>
          </div>
        ) : null}
        {lookup && (
          <PatientDocumentsModal
            mrn={lookup.mrn}
            open={docsOpen}
            onClose={() => { setDocsOpen(false); void handleLookup(); }}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}