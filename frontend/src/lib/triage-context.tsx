/* eslint-disable prettier/prettier */
/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useContext, useState, useMemo, useEffect } from "react";

export type EsiLevel = 1 | 2 | 3 | 4 | 5;

export interface VitalRecord {
  hr?: number | null;
  bpSys?: number | null;
  bpDia?: number | null;
  spo2?: number | null;
  temp?: number | null;
  timestamp: string;
}

export interface Patient {
  id: string;
  name: string;
  mrn: string;
  age: number;
  gender: "M" | "F" | "O" | "";
  complaint: string;
  esi: EsiLevel;
  waitMins: number;
  alert?: boolean;
  addedAt?: number;
  vitalsHistory: VitalRecord[];
  aiConfidence?: number;
}

export interface ArchivedPatient extends Patient {
  disposition: string;
  dispositionTime: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  auditLog: any;
  isAnonymized?: boolean;
}

export const BASE_PATIENTS: Patient[] = [
  { id: "p1", name: "Doe, John", mrn: "MRN-A3F9", age: 45, gender: "M", complaint: "Crushing chest pain radiating to left arm, diaphoretic", esi: 1, waitMins: 0, vitalsHistory: [], aiConfidence: 94 },
  { id: "p2", name: "Okafor, Amara", mrn: "MRN-B7K2", age: 62, gender: "F", complaint: "Unresponsive episode, GCS 13, post-ictal", esi: 1, waitMins: 3, vitalsHistory: [], aiConfidence: 89 },
  { id: "p3", name: "Reyes, Miguel", mrn: "MRN-C2M8", age: 58, gender: "M", complaint: "Acute shortness of breath, SpO2 89% on room air", esi: 2, waitMins: 5, vitalsHistory: [], aiConfidence: 91 },
  { id: "p4", name: "Chen, Li Wei", mrn: "MRN-D9P4", age: 34, gender: "F", complaint: "Severe abdominal pain RLQ, rebound tenderness", esi: 2, waitMins: 12, vitalsHistory: [], aiConfidence: 86 },
  { id: "p5", name: "Patel, Priya", mrn: "MRN-E4R7", age: 41, gender: "F", complaint: "Suspected DVT, unilateral leg swelling and pain", esi: 2, waitMins: 22, vitalsHistory: [], aiConfidence: 82 },
  { id: "p6", name: "Smith, Karen", mrn: "MRN-F6T1", age: 29, gender: "F", complaint: "Abdominal pain, nausea, vitals stable", esi: 3, waitMins: 40, alert: true, vitalsHistory: [], aiConfidence: 78 },
  { id: "p7", name: "Novak, Tomas", mrn: "MRN-G8W5", age: 53, gender: "M", complaint: "Kidney stone, flank pain 8/10, hematuria", esi: 3, waitMins: 47, vitalsHistory: [], aiConfidence: 95 },
  { id: "p8", name: "Garcia, Sofia", mrn: "MRN-H1Y9", age: 24, gender: "F", complaint: "Minor forearm laceration, bleeding controlled", esi: 4, waitMins: 15, vitalsHistory: [], aiConfidence: 88 },
  { id: "p9", name: "Kim, Daniel", mrn: "MRN-J5Q3", age: 38, gender: "M", complaint: "Sprained ankle, ambulatory, no deformity", esi: 4, waitMins: 31, vitalsHistory: [], aiConfidence: 92 },
  { id: "p10", name: "Brown, Alice", mrn: "MRN-K3V6", age: 67, gender: "F", complaint: "Prescription refill request, asymptomatic", esi: 5, waitMins: 10, vitalsHistory: [], aiConfidence: 99 },
  { id: "p11", name: "Ibrahim, Yusuf", mrn: "MRN-L8N2", age: 19, gender: "M", complaint: "Sore throat, low-grade fever x2 days", esi: 5, waitMins: 26, vitalsHistory: [], aiConfidence: 85 },
];

export const SURGE_PATIENTS: Patient[] = [
  { id: "s1", name: "Gill, Hanna", mrn: "MRN-M2X8", age: 71, gender: "F", complaint: "Multi-vehicle MVC, chest trauma, hypotensive", esi: 1, waitMins: 1, vitalsHistory: [], aiConfidence: 96 },
  { id: "s2", name: "Fischer, Otto", mrn: "MRN-N6C4", age: 55, gender: "M", complaint: "MVC walk-in, head laceration, LOC at scene", esi: 2, waitMins: 4, vitalsHistory: [], aiConfidence: 91 },
  { id: "s3", name: "Adeyemi, Femi", mrn: "MRN-P9Z1", age: 30, gender: "M", complaint: "MVC, wrist deformity, stable vitals", esi: 3, waitMins: 6, vitalsHistory: [], aiConfidence: 84 },
  { id: "s4", name: "Ruiz, Carlos", mrn: "MRN-X1L2", age: 48, gender: "M", complaint: "Chemical burn to face and eyes, severe pain", esi: 1, waitMins: 0, vitalsHistory: [], aiConfidence: 95 },
  { id: "s5", name: "Chang, Wei", mrn: "MRN-Y4T7", age: 65, gender: "M", complaint: "Sudden onset left-sided weakness, slurred speech", esi: 1, waitMins: 2, vitalsHistory: [], aiConfidence: 92 },
  { id: "s6", name: "Mwangi, Grace", mrn: "MRN-Z8Q3", age: 22, gender: "F", complaint: "Asthma exacerbation, audible wheezing, SpO2 91%", esi: 2, waitMins: 8, vitalsHistory: [], aiConfidence: 88 },
  { id: "s7", name: "Ivanov, Alexei", mrn: "MRN-W5R9", age: 40, gender: "M", complaint: "Workplace crush injury to lower leg, diminished pulses", esi: 2, waitMins: 11, vitalsHistory: [], aiConfidence: 86 },
  { id: "s8", name: "Singh, Arjun", mrn: "MRN-V2P4", age: 28, gender: "M", complaint: "High fever 39.5C, stiff neck, photophobia", esi: 2, waitMins: 14, vitalsHistory: [], aiConfidence: 89 },
  { id: "s9", name: "Dubois, Claire", mrn: "MRN-U7K1", age: 35, gender: "F", complaint: "Severe migraine, vomiting, prior hx of migraines", esi: 3, waitMins: 18, vitalsHistory: [], aiConfidence: 93 },
  { id: "s10", name: "Monalisa, Liam", mrn: "MRN-T3M6", age: 50, gender: "M", complaint: "Deep laceration to thigh, bleeding controlled with pressure", esi: 3, waitMins: 25, vitalsHistory: [], aiConfidence: 90 },
];

export const ESI_META: Record<EsiLevel, { label: string; badge: string; dot: string; swatch: string }> = {
  1: { label: "Immediate", badge: "bg-esi-1 text-primary-foreground", dot: "bg-esi-1", swatch: "var(--esi-1)" },
  2: { label: "Emergent", badge: "bg-esi-2 text-primary-foreground", dot: "bg-esi-2", swatch: "var(--esi-2)" },
  3: { label: "Urgent", badge: "bg-esi-3 text-foreground", dot: "bg-esi-3", swatch: "var(--esi-3)" },
  4: { label: "Less Urgent", badge: "bg-esi-4 text-primary-foreground", dot: "bg-esi-4", swatch: "var(--esi-4)" },
  5: { label: "Non-Urgent", badge: "bg-esi-5 text-primary-foreground", dot: "bg-esi-5", swatch: "var(--esi-5)" },
};

export function formatWait(mins: number) {
  if (mins < 60) return `${mins} mins`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

export function checkDeterioration(patient: Patient, waitOffset: number): boolean {
  if (patient.alert) return true;
  const effectiveWait = patient.waitMins + waitOffset;
  if (patient.esi === 1 && effectiveWait >= 5) return true; // ESI 1 alerts after 5 mins
  if (patient.esi === 2 && effectiveWait >= 15) return true;
  if (patient.esi === 3 && effectiveWait >= 60) return true;
  return false;
}

interface TriageContextState {
  aiActive: boolean;
  setAiActive: (val: boolean) => void;
  surge: boolean;
  setSurge: (val: boolean) => void;
  waitOffset: number;
  setWaitOffset: (val: number) => void;
  sortMode: "wait" | "priority";
  setSortMode: (val: "wait" | "priority") => void;
  searchQuery: string;
  setSearchQuery: (val: string) => void;
  patients: Patient[];
  allPatientsCount: number;
  isAddModalOpen: boolean;
  setIsAddModalOpen: (val: boolean) => void;
  updateVitalsModal: { isOpen: boolean; patientId: string | null };
  setUpdateVitalsModal: (state: { isOpen: boolean; patientId: string | null }) => void;
  dispositionModal: { isOpen: boolean; patientId: string | null };
  setDispositionModal: (state: { isOpen: boolean; patientId: string | null }) => void;
  addPatient: (patient: Omit<Patient, "id" | "vitalsHistory">) => void;
  updatePatientVitals: (id: string, newEsi: EsiLevel, newVitals: Omit<VitalRecord, "timestamp">) => void;
  
  // FIX: Explicitly match the Record<string, unknown> implementation below
  removePatient: (id: string, disposition: string, log: Record<string, unknown>) => void;
  
  archivedPatients: ArchivedPatient[];
  revertPatient: (id: string) => void;
  anonymizePatient: (id: string) => void;
}

const TriageContext = createContext<TriageContextState | undefined>(undefined);

export function TriageProvider({ children }: { children: React.ReactNode }) {
  const [aiActive, setAiActive] = useState(true);
  const [surge, setSurge] = useState(false);
  const [waitOffset, setWaitOffset] = useState(0);
  const [sortMode, setSortMode] = useState<"wait" | "priority">("priority");
  const [searchQuery, setSearchQuery] = useState("");
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [updateVitalsModal, setUpdateVitalsModal] = useState<{ isOpen: boolean; patientId: string | null }>({ isOpen: false, patientId: null });
  const [dispositionModal, setDispositionModal] = useState<{ isOpen: boolean; patientId: string | null }>({ isOpen: false, patientId: null });
  
  const [baseList, setBaseList] = useState<Patient[]>(() => BASE_PATIENTS.map((p) => ({ ...p, addedAt: 0 })));
  const [surgeList, setSurgeList] = useState<Patient[]>([]);
  const [archivedPatients, setArchivedPatients] = useState<ArchivedPatient[]>([]);

  useEffect(() => {
    if (surge) {
      const now = Date.now();
      setSurgeList(SURGE_PATIENTS.map((p) => ({ ...p, addedAt: now })));
    } else {
      setSurgeList([]);
    }
  }, [surge]);

  const addPatient = (patientData: Omit<Patient, "id" | "vitalsHistory">) => {
    const newPatient: Patient = {
      ...patientData,
      id: Math.random().toString(36).substr(2, 9),
      vitalsHistory: [],
      addedAt: Date.now()
    };
    setBaseList((prev) => [newPatient, ...prev]);
  };

  const updatePatientVitals = (id: string, newEsi: EsiLevel, newVitals: Omit<VitalRecord, "timestamp">) => {
    setBaseList((prev) => prev.map((p) => {
      if (p.id === id) {
        return {
          ...p,
          esi: newEsi,
          vitalsHistory: [...p.vitalsHistory, { ...newVitals, timestamp: new Date().toISOString() }],
        };
      }
      return p;
    }));
    setSurgeList((prev) => prev.map((p) => {
      if (p.id === id) {
        return {
          ...p,
          esi: newEsi,
          vitalsHistory: [...p.vitalsHistory, { ...newVitals, timestamp: new Date().toISOString() }],
        };
      }
      return p;
    }));
  };

  const removePatient = (id: string, disposition: string, log: Record<string, unknown>) => {
    console.log("DISPOSITION AUDIT LOG:", { id, disposition, ...log });
    const patientToArchive = baseList.find(p => p.id === id) || surgeList.find(p => p.id === id);
    if (patientToArchive) {
      const archivedRecord: ArchivedPatient = {
        ...patientToArchive,
        disposition,
        dispositionTime: new Date().toISOString(),
        auditLog: log
      };
      setArchivedPatients(prev => [archivedRecord, ...prev]);
    }
    setBaseList((prev) => prev.filter((p) => p.id !== id));
    setSurgeList((prev) => prev.filter((p) => p.id !== id));
  };

  const revertPatient = (id: string) => {
    const patient = archivedPatients.find(p => p.id === id);
    if (!patient) return;
    const dispTime = new Date(patient.dispositionTime).getTime();
    const minutesSinceDisposition = (Date.now() - dispTime) / (1000 * 60);
    if (minutesSinceDisposition > 15) {
      alert("Compliance Error: Cannot revert. The 15-minute window has expired.");
      return;
    }
    setArchivedPatients(prev => prev.filter(p => p.id !== id));
    const { disposition, dispositionTime, auditLog, isAnonymized, ...activePatient } = patient;
    setBaseList(prev => [activePatient, ...prev]);
  };

  const anonymizePatient = (id: string) => {
    setArchivedPatients(prev => prev.map(p => {
      if (p.id === id) {
        return { ...p, name: "REDACTED", mrn: "REDACTED_PII", isAnonymized: true };
      }
      return p;
    }));
  };

  const patients = useMemo(() => {
    const queue = [...baseList, ...surgeList];
    const filtered = queue.filter((p) => {
      if (!searchQuery) return true;
      const q = searchQuery.toLowerCase();
      return (
        p.name.toLowerCase().includes(q) ||
        p.mrn.toLowerCase().includes(q) ||
        p.complaint.toLowerCase().includes(q)
      );
    });
    return filtered.sort((a, b) => sortMode === "wait" ? (b.waitMins + waitOffset) - (a.waitMins + waitOffset) : a.esi - b.esi || (b.waitMins + waitOffset) - (a.waitMins + waitOffset));
  }, [baseList, surgeList, sortMode, waitOffset, searchQuery]);

  const allPatientsCount = baseList.length + surgeList.length;

  return (
    <TriageContext.Provider value={{
      aiActive, setAiActive, surge, setSurge, waitOffset, setWaitOffset, sortMode, setSortMode,
      searchQuery, setSearchQuery, patients, allPatientsCount, isAddModalOpen, setIsAddModalOpen,
      updateVitalsModal, setUpdateVitalsModal, dispositionModal, setDispositionModal,
      addPatient, updatePatientVitals, removePatient, archivedPatients, revertPatient, anonymizePatient
    }}>
      {children}
    </TriageContext.Provider>
  );
}

export function useTriage() {
  const context = useContext(TriageContext);
  if (context === undefined) {
    throw new Error("useTriage must be used within a TriageProvider");
  }
  return context;
}