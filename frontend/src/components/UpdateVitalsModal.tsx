/* eslint-disable prettier/prettier */
import React, { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { useTriage, EsiLevel } from "../lib/triage-context";

export function UpdateVitalsModal() {
  const { patients, updateVitalsModal, setUpdateVitalsModal, updatePatientVitals } = useTriage();
  const [isLoading, setIsLoading] = useState(false);
  
  const [hr, setHr] = useState("");
  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [spo2, setSpo2] = useState("");
  const [temp, setTemp] = useState("");
  
  const [aiRecommendation, setAiRecommendation] = useState<EsiLevel | null>(null);
  const [isDowngrade, setIsDowngrade] = useState(false);
  
  const patient = patients.find(p => p.id === updateVitalsModal.patientId);
  
  useEffect(() => {
    if (!updateVitalsModal.isOpen) {
      setHr("");
      setBpSys("");
      setBpDia("");
      setSpo2("");
      setTemp("");
      setAiRecommendation(null);
      setIsDowngrade(false);
    }
  }, [updateVitalsModal.isOpen]);
  
  if (!patient) return null;

  // Helper to parse string inputs into strict numbers to satisfy the VitalRecord type
  const getParsedVitals = () => ({
    hr: parseInt(hr) || null,
    bpSys: parseInt(bpSys) || null,
    bpDia: parseInt(bpDia) || null,
    spo2: parseInt(spo2) || null,
    temp: parseFloat(temp) || null,
  });

  const handleRunReTriage = async () => {
    const numHr = parseInt(hr);
    if (numHr > 220 || numHr < 30) {
      if (!window.confirm(`WARNING: Heart rate of ${numHr} is extremely abnormal. Are you sure this is not a typo?`)) {
        return;
      }
    }
    
    setIsLoading(true);
    const parsedVitals = getParsedVitals();
    
    try {
      const payload = {
        mrn: patient.mrn,
        name: patient.name,
        age: patient.age,
        complaint: patient.complaint,
        ...parsedVitals
      };
      
      const API_URL = import.meta.env['VITE_API_URL'] || "http://127.0.0.1:8000";
      const response = await fetch(`${API_URL}/api/triage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const data = await response.json();
      const newEsi = data.recommended_esi as EsiLevel;
      
      // Asymmetric Cost Logic: Check if AI is trying to downgrade acuity
      if (newEsi > patient.esi) {
        setAiRecommendation(newEsi);
        setIsDowngrade(true);
      } else {
        updatePatientVitals(patient.id, newEsi, parsedVitals);
        setUpdateVitalsModal({ isOpen: false, patientId: null });
      }
    } catch (e) {
      console.error(e);
      updatePatientVitals(patient.id, patient.esi, parsedVitals);
      setUpdateVitalsModal({ isOpen: false, patientId: null });
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <Dialog open={updateVitalsModal.isOpen} onOpenChange={(open) => setUpdateVitalsModal({ isOpen: open, patientId: null })}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Update Vitals: {patient.name}</DialogTitle>
        </DialogHeader>
        {!isDowngrade ? (
          <div className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-bold text-slate-500">Heart Rate</label>
                <input type="number" value={hr} onChange={e => setHr(e.target.value)} className="w-full p-2 border rounded" placeholder="BPM" />
              </div>
              <div>
                <label className="text-xs font-bold text-slate-500">SpO2 %</label>
                <input type="number" value={spo2} onChange={e => setSpo2(e.target.value)} className="w-full p-2 border rounded" placeholder="%" />
              </div>
              <div>
                <label className="text-xs font-bold text-slate-500">BP Systolic</label>
                <input type="number" value={bpSys} onChange={e => setBpSys(e.target.value)} className="w-full p-2 border rounded" placeholder="mmHg" />
              </div>
              <div>
                <label className="text-xs font-bold text-slate-500">Temperature (°C)</label>
                <input type="number" step="0.1" value={temp} onChange={e => setTemp(e.target.value)} className="w-full p-2 border rounded" placeholder="°C" />
              </div>
            </div>
            <button onClick={handleRunReTriage} disabled={isLoading} className="w-full py-2 bg-blue-600 text-white font-bold rounded hover:bg-blue-700">
              {isLoading ? "Running Delta Engine..." : "Submit & Re-Triage"}
            </button>
          </div>
        ) : (
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg space-y-4">
            <div className="flex items-start gap-2">
              <span className="text-2xl">⚠️</span>
              <div>
                <h4 className="font-bold text-yellow-800">Down-Triage Detected</h4>
                <p className="text-sm text-yellow-700">
                  The AI suggests downgrading this patient from <strong>ESI {patient.esi}</strong> to <strong>ESI {aiRecommendation}</strong> based on the improved vitals.
                </p>
                <p className="text-xs text-yellow-600 mt-2 italic">
                  Hospital policy requires explicit clinical confirmation to lower an assigned ESI level.
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => { updatePatientVitals(patient.id, aiRecommendation!, getParsedVitals()); setUpdateVitalsModal({ isOpen: false, patientId: null }); }} className="flex-1 py-2 bg-yellow-600 text-white font-bold rounded hover:bg-yellow-700">
                Confirm Downgrade
              </button>
              <button onClick={() => { updatePatientVitals(patient.id, patient.esi, getParsedVitals()); setUpdateVitalsModal({ isOpen: false, patientId: null }); }} className="flex-1 py-2 bg-slate-200 text-slate-800 font-bold rounded hover:bg-slate-300">
                Keep ESI {patient.esi}
              </button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}