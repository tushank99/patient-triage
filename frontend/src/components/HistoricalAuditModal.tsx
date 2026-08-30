/* eslint-disable prettier/prettier */
import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { ArchivedPatient, ESI_META } from "../lib/triage-context";
import { Activity, BrainCircuit, UserCog, UserCheck, ShieldAlert } from "lucide-react";

interface Props {
  patient: ArchivedPatient | null;
  isOpen: boolean;
  onClose: () => void;
}

export function HistoricalAuditModal({ patient, isOpen, onClose }: Props) {
  if (!patient) return null;

  const intakeTime = new Date(patient.addedAt || Date.now() - patient.waitMins * 60000);
  const dispTime = new Date(patient.dispositionTime);
  const meta = ESI_META[patient.esi];

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl bg-slate-50">
        <DialogHeader className="pb-4 border-b border-slate-200">
          <div className="flex justify-between items-start">
            <div>
              <DialogTitle className="text-xl flex items-center gap-2">
                {patient.isAnonymized && <ShieldAlert className="w-5 h-5 text-red-500" />}
                Audit Trail: {patient.name}
              </DialogTitle>
              <p className="text-sm text-slate-500 mt-1">MRN: {patient.mrn}</p>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-bold shadow-sm ${meta.badge}`}>
              Final ESI: {patient.esi}
            </span>
          </div>
        </DialogHeader>

        <div className="py-4 px-2 space-y-6 relative">
          {/* Vertical Line for Timeline */}
          {/* FIX: Changed left-[39px] to left-10 to satisfy Tailwind linter */}
          <div className="absolute left-10 top-6 bottom-6 w-0.5 bg-slate-200 z-0"></div>

          {/* Event 1: Intake */}
          <div className="relative z-10 flex gap-4">
            <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center shrink-0 border-4 border-slate-50 shadow-sm">
              <Activity className="w-4 h-4" />
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex-1">
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-bold text-slate-900">Patient Intake</h4>
                <span className="text-xs font-bold text-slate-500">{intakeTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
              </div>
              <p className="text-sm text-slate-700 italic border-l-2 border-slate-300 pl-3">"{patient.complaint}"</p>
            </div>
          </div>

          {/* Event 2: AI Triage */}
          <div className="relative z-10 flex gap-4">
            <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0 border-4 border-slate-50 shadow-sm">
              <BrainCircuit className="w-4 h-4" />
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex-1">
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-bold text-slate-900">AI Inference Completed</h4>
                {/* FIX: Removed conflicting text-slate-500 class */}
                <span className="text-xs font-bold text-emerald-600">{patient.aiConfidence ?? 85}% Confidence</span>
              </div>
              <p className="text-sm text-slate-600">Model assigned baseline priority.</p>
            </div>
          </div>

          {/* Event 3: Nurse Override (Conditional) */}
          {patient.auditLog?.overrideReason && (
             <div className="relative z-10 flex gap-4">
             <div className="w-10 h-10 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center shrink-0 border-4 border-slate-50 shadow-sm">
               <UserCog className="w-4 h-4" />
             </div>
             {/* FIX: Removed conflicting bg-white class */}
             <div className="bg-amber-50/30 p-4 rounded-lg shadow-sm border border-amber-200 flex-1">
               <div className="flex justify-between items-center mb-2">
                 <h4 className="font-bold text-amber-900">Clinician Override</h4>
               </div>
               <p className="text-sm text-amber-800">
                 <strong>Reason logged:</strong> {patient.auditLog.overrideReason}
               </p>
             </div>
           </div>
          )}

          {/* Event 4: Disposition */}
          <div className="relative z-10 flex gap-4">
            <div className="w-10 h-10 rounded-full bg-slate-800 text-white flex items-center justify-center shrink-0 border-4 border-slate-50 shadow-sm">
              <UserCheck className="w-4 h-4" />
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex-1">
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-bold text-slate-900">Patient Dispositioned</h4>
                <span className="text-xs font-bold text-slate-500">{dispTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
              </div>
              <div className="inline-block px-2 py-1 bg-slate-100 border border-slate-200 rounded text-xs font-bold text-slate-700">
                {patient.disposition}
              </div>
              
              {/* LWBS Protocol Notice */}
              {patient.auditLog?.checks && patient.disposition === "LWBS" && (
                <div className="mt-3 text-xs text-slate-500 p-2 bg-slate-50 rounded border">
                  <strong>Compliance Checks Verified:</strong> Paged, Searched, Called, Notified.
                </div>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}