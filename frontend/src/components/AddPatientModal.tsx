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

export function AddPatientModal() {
  const { isAddModalOpen, setIsAddModalOpen, addPatient } = useTriage();
  
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
  const [history, setHistory] = useState(false);

  // AI & Session State
  const [currentMrn, setCurrentMrn] = useState("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [aiResult, setAiResult] = useState<any>(null);
  
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
    setHistory(false);
    
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
  const numSpo2 = spo2 ? parseInt(spo2, 10) : null;

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

  const hasErrors = errTemp || errHr || errSpo2 || errBpSys || errBpDia || errBpLogic;
  const hasMandatory = name.trim() !== "" && gender !== "" && dob !== "" && complaint.trim() !== "";
  const canProceed = hasMandatory && !hasErrors;

  const skippedVitals = [];
  if (!temp) skippedVitals.push("Temperature");
  if (!bpSys || !bpDia) skippedVitals.push("Blood Pressure");
  if (!hr) skippedVitals.push("Heart Rate");
  if (!spo2) skippedVitals.push("Oxygen Saturation");
  if (!pain) skippedVitals.push("Pain Score");

  const handleSubmit = async () => {
    setIsLoading(true);
    const generatedMrn = `MRN-WALK-${Math.floor(100 + Math.random() * 900)}`;
    setCurrentMrn(generatedMrn);

    const payload = {
      mrn: history ? generatedMrn : null,
      age_years: age !== null ? Math.floor(age) : 0,
      gender: gender,
      chief_complaint: complaint,
      has_prior_history: history,
      vitals: {
        heart_rate: numHr,
        systolic_bp: numBpSys,
        diastolic_bp: numBpDia,
        oxygen_saturation: numSpo2,
        temperature_celsius: normalizedTemp,
        pain_score: pain ? parseInt(pain, 10) : null
      }
    };

    try {
      const API_URL = import.meta.env['VITE_API_URL'] || "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/api/triage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Backend triage failed");
      
      const data = await response.json();
      setAiResult(data);
      setStep(3);
      
    } catch (err) {
      console.error("API Error. Circuit-breaking to local ONNX model:", err);
      try {
        const edgeResult = await runEdgeInference({
          age,
          hr: numHr,
          bpSys: numBpSys,
          bpDia: numBpDia,
          spo2: numSpo2,
          temp: normalizedTemp,
          pain: null,
          hasMrn: false,
        });
        setAiResult(edgeResult);
      } catch (onnxErr) {
        console.error("ONNX fallback also failed:", onnxErr);
        setAiResult({
          recommended_esi: 3 as EsiLevel,
          confidence_score: 40,
          rationale: "SYSTEM OFFLINE: Both cloud API and local ONNX model unavailable. Defaulting to ESI 3 (safety-first escalation).",
          badges: [{ label: " Full Offline", type: "danger" }],
        });
      }
      setStep(3);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFinalizeTriage = (finalEsi: EsiLevel) => {
    addPatient({
      name,
      mrn: currentMrn,
      age: age !== null ? Math.floor(age) : 0,
      gender: gender as "M" | "F",
      complaint,
      esi: finalEsi,
      waitMins: 0,
      aiConfidence: aiResult?.confidence_score
    });
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
                  
                  {/* Hx */}
                  <div className="flex items-center h-full pt-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={history}
                        onChange={(e) => setHistory(e.target.checked)}
                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm font-medium text-foreground">Prior Medical History Available</span>
                    </label>
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
                      <p className="text-xs text-muted-foreground">Prior History</p>
                      <p className="font-semibold text-foreground">{history ? "Available" : "None Indicated"}</p>
                    </div>
                  </div>
                </div>
              </div>

            </div>
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
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-sm font-bold uppercase text-muted-foreground">Confidence Score</h3>
                      <span className={`text-lg font-black ${aiResult.confidence_score >= 85 ? 'text-green-600' : aiResult.confidence_score >= 70 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {aiResult.confidence_score}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                      <div 
                        className={`h-2.5 rounded-full ${aiResult.confidence_score >= 85 ? 'bg-green-500' : aiResult.confidence_score >= 70 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${aiResult.confidence_score}%` }}
                      ></div>
                    </div>
                    
                    {/* Deterministic Badges */}
                    {aiResult.badges && aiResult.badges.length > 0 && (
                      <div className="mt-5">
                        <h4 className="text-xs font-bold uppercase text-muted-foreground mb-2">Clinical Flags</h4>
                        <div className="flex flex-wrap gap-2">
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          {aiResult.badges.map((badge: any, i: number) => (
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
                        onClick={() => {
                          console.log("AUDIT LOG EVENT:", {
                            mrn: currentMrn,
                            ai_recommended_esi: aiResult.recommended_esi,
                            human_overridden_esi: selectedOverrideEsi,
                            reason: overrideReason,
                            timestamp: new Date().toISOString()
                          });
                          handleFinalizeTriage(selectedOverrideEsi);
                        }}
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
            <div className="p-6 border-t border-border bg-white flex justify-between items-center">
              <button
                type="button"
                onClick={() => setIsOverriding(true)}
                disabled={isOverriding}
                className="text-sm font-semibold text-slate-600 hover:text-red-600 transition-colors disabled:opacity-50"
              >
                Manual Override
              </button>
              <button
                type="button"
                onClick={() => handleFinalizeTriage(aiResult.recommended_esi as EsiLevel)}
                disabled={isOverriding}
                className="rounded-md bg-green-600 px-6 py-2.5 text-sm font-bold text-white shadow-sm transition-colors hover:bg-green-700 disabled:opacity-50"
              >
                Accept AI Decision
              </button>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}