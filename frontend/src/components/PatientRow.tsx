/* eslint-disable prettier/prettier */
import { AlertTriangle, Activity, UserMinus, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { ESI_META, Patient, checkDeterioration, formatWait, useTriage } from "../lib/triage-context";
import { useAuth } from "../lib/auth";


export function PatientRow({ patient }: { patient: Patient }) {
  const { aiActive, waitOffset, setUpdateVitalsModal, setDispositionModal, acknowledgeReview } = useTriage();
  const { can } = useAuth();
  const canAct = can("triage:write");
  const meta = ESI_META[patient.esi];
  const displayedWait = patient.waitMins + waitOffset;
  const isDeteriorated = checkDeterioration(patient, waitOffset);

  const [isNew, setIsNew] = useState(false);

  useEffect(() => {
    if (patient.addedAt && Date.now() - patient.addedAt < 3000) {
      setIsNew(true);
      const timer = setTimeout(() => setIsNew(false), 3000);
      return () => clearTimeout(timer);
    }

    return undefined;
  }, [patient.addedAt]);

  return (
    <div
      className={`grid grid-cols-[minmax(0,1.6fr)_70px_minmax(0,2.2fr)_170px_140px] items-center gap-4 border-b border-border px-5 py-4 transition-colors last:border-b-0 hover:bg-muted/40 ${
        isDeteriorated ? "border-l-4 border-l-esi-1 bg-alert-bg" : "border-l-4 border-l-transparent"
      } ${isNew ? "animate-new-row" : ""}`}
    >
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold text-foreground">{patient.name}</p>
        <p className="text-xs text-muted-foreground">{patient.mrn}</p>
      </div>
      <p className="text-sm text-muted-foreground">
        {patient.age} {patient.gender}
      </p>
      <div className="min-w-0">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <p className="truncate text-sm text-foreground cursor-default hover:underline decoration-dashed underline-offset-4 decoration-muted-foreground/50">
                {patient.complaint}
              </p>
            </TooltipTrigger>
            {/* FIX: Changed break-words to wrap-break-word per Tailwind suggestion */}
            <TooltipContent side="top" className="max-w-xs wrap-break-word">
              <p>{patient.complaint}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {patient.needsReview && (
          <div className="mt-1 flex items-center gap-2">
            <span title={patient.reviewReasons.join("\n")} className="inline-flex items-center gap-1 rounded-md bg-amber-100 px-2 py-0.5 text-[11px] font-bold text-amber-800">
              <ShieldCheck className="h-3 w-3" /> Review needed
            </span>
            {canAct && (
              <button type="button" onClick={() => void acknowledgeReview(patient.id)} className="text-[11px] font-semibold text-blue-600 hover:underline">
                Mark reviewed
              </button>
            )}
          </div>
        )}

        {isDeteriorated && (
          <span className="mt-1 inline-flex items-center gap-1 rounded-md bg-esi-1-soft px-2 py-0.5 text-[11px] font-bold text-esi-1">
            <AlertTriangle className="h-3 w-3" />
            Needs Reassessment — waited {formatWait(displayedWait)}
          </span>
        )}
      </div>

      <div>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold ${meta.badge}`}
        >
          ESI {patient.esi} · {meta.label}
        </span>
        {aiActive && (
          <p className="mt-1 text-[11px] text-muted-foreground font-medium">
            Model probability: <span className={`font-bold ${patient.aiConfidence === undefined ? 'text-muted-foreground' : patient.aiConfidence < 80 ? 'text-amber-600' : 'text-emerald-600'}`}>
              {patient.aiConfidence === undefined ? "—" : `${patient.aiConfidence}%`}
            </span>
          </p>
        )}
      </div>

      <div className="flex flex-col items-end gap-1">
        <p className={`text-sm font-semibold ${displayedWait >= 45 ? "text-esi-1" : "text-foreground"} mb-1`}>
          {formatWait(displayedWait)}
        </p>
        <div className="flex gap-1">
          {canAct && <button
            onClick={() => setUpdateVitalsModal({ isOpen: true, patientId: patient.id })}
            title="Update Vitals"
            className="p-1.5 text-blue-600 bg-blue-50 hover:bg-blue-100 rounded transition-colors"
          >
            <Activity className="h-4 w-4" />
          </button>}
          {canAct && <button
            onClick={() => setDispositionModal({ isOpen: true, patientId: patient.id })}
            title="Discharge Patient"
            className="p-1.5 text-slate-600 bg-slate-100 hover:bg-slate-200 rounded transition-colors"
          >
            <UserMinus className="h-4 w-4" />
          </button>}
        </div>
      </div>
    </div>
  );
}