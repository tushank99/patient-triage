/* eslint-disable prettier/prettier */
import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { useTriage } from "../lib/triage-context";

export function DispositionModal() {
  const { patients, dispositionModal, setDispositionModal, removePatient } = useTriage();
  
  const [disposition, setDisposition] = useState("");
  const [checks, setChecks] = useState({ paged: false, searched: false, called: false, notified: false });
  const [showConfirm, setShowConfirm] = useState(false); // New state for custom confirmation UI

  const patient = patients.find(p => p.id === dispositionModal.patientId);
  if (!patient) return null;

  const isHighRiskLwbs = disposition === "LWBS" && (patient.esi === 1 || patient.esi === 2 || patient.esi === 3);
  const isLwbsCompliant = checks.paged && checks.searched && checks.called && checks.notified;
  const canSubmit = disposition && (!isHighRiskLwbs || isLwbsCompliant);

  const resetModalState = () => {
    setDispositionModal({ isOpen: false, patientId: null });
    setDisposition("");
    setShowConfirm(false);
    setChecks({ paged: false, searched: false, called: false, notified: false });
  };

  const handleFinalConfirm = () => {
    removePatient(patient.id, disposition, { checks, timestamp: new Date().toISOString() });
    resetModalState();
  };

  // If the user clicks confirm on the first screen, show the warning screen
  if (showConfirm) {
    return (
      <Dialog open={dispositionModal.isOpen} onOpenChange={(open) => !open && resetModalState()}>
        <DialogContent className="max-w-md border-red-200">
          <DialogHeader>
            <DialogTitle className="text-red-700 flex items-center gap-2">
              <span className="text-xl">⚠️</span> Confirm Patient Removal
            </DialogTitle>
          </DialogHeader>
          <div className="py-4 space-y-3">
            <p className="text-slate-700">
              Are you sure you want to permanently remove <strong className="text-slate-900">{patient.name}</strong> (ESI {patient.esi}) from the active triage queue?
            </p>
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-md">
              <p className="text-xs text-slate-500 uppercase font-bold tracking-wider mb-1">Selected Disposition</p>
              <p className="font-semibold text-slate-900">{disposition}</p>
            </div>
            {isHighRiskLwbs && (
              <p className="text-xs text-red-600 font-medium">
                * Note: High-acuity LWBS protocol checklist will be logged to the audit trail.
              </p>
            )}
          </div>
          <div className="flex gap-3 mt-2">
            <button 
              onClick={() => setShowConfirm(false)}
              className="flex-1 py-2 bg-slate-100 text-slate-700 font-bold rounded-lg hover:bg-slate-200 transition-colors"
            >
              Cancel
            </button>
            <button 
              onClick={handleFinalConfirm}
              className="flex-1 py-2 bg-red-600 text-white font-bold rounded-lg hover:bg-red-700 transition-colors shadow-sm"
            >
              Confirm Removal
            </button>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // Default Selection Screen
  return (
    <Dialog open={dispositionModal.isOpen} onOpenChange={(open) => !open && resetModalState()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Discharge / Remove Patient</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <p className="text-sm text-slate-500 mb-2">Select outcome for <strong>{patient.name}</strong> (ESI {patient.esi}):</p>
          
          <div className="space-y-2">
            {["Admitted to ER Bed", "Discharged / Re-routed", "LWBS"].map(opt => (
              <button
                key={opt}
                onClick={() => setDisposition(opt)}
                className={`w-full text-left px-4 py-3 border rounded-lg font-medium transition-colors ${
                  disposition === opt ? 'bg-blue-50 border-blue-600 text-blue-700' : 'bg-white hover:bg-slate-50'
                }`}
              >
                {opt === "LWBS" ? "Left Without Being Seen (LWBS)" : opt}
              </button>
            ))}
          </div>

          {isHighRiskLwbs && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg animate-in slide-in-from-top-2">
              <h4 className="font-bold text-red-800 text-sm mb-2 flex items-center gap-1">
                <span>🛑</span> High-Acuity LWBS Protocol Required
              </h4>
              <p className="text-xs text-red-700 mb-3">
                This patient is classified as high-risk (ESI {patient.esi}). To mitigate hospital liability, you must document completion of the search protocol before removing them from the board.
              </p>
              
              <div className="space-y-2 text-sm text-red-900 font-medium">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={checks.paged} onChange={e => setChecks({...checks, paged: e.target.checked})} className="accent-red-600" />
                  Paged patient overhead 3 times
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={checks.searched} onChange={e => setChecks({...checks, searched: e.target.checked})} className="accent-red-600" />
                  Physical search of waiting area & restrooms
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={checks.called} onChange={e => setChecks({...checks, called: e.target.checked})} className="accent-red-600" />
                  Attempted contact via phone on file
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={checks.notified} onChange={e => setChecks({...checks, notified: e.target.checked})} className="accent-red-600" />
                  Notified Charge Nurse / Attending
                </label>
              </div>
            </div>
          )}
        </div>

        <button 
          onClick={() => setShowConfirm(true)} 
          disabled={!canSubmit}
          className="w-full py-2 bg-slate-900 text-white font-bold rounded-lg hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Review Disposition
        </button>
      </DialogContent>
    </Dialog>
  );
}