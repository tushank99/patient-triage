import React, { createContext, useContext, useState, useMemo, useEffect } from "react";

export type EsiLevel = 1 | 2 | 3 | 4 | 5;

export interface Patient {
  name: string;
  mrn: string;
  age: number;
  gender: "M" | "F";
  complaint: string;
  esi: EsiLevel;
  waitMins: number;
  alert?: boolean;
  addedAt?: number;
}

export const BASE_PATIENTS: Patient[] = [
  {
    name: "Doe, John",
    mrn: "MRN-A3F9",
    age: 45,
    gender: "M",
    complaint: "Crushing chest pain radiating to left arm, diaphoretic",
    esi: 1,
    waitMins: 0,
  },
  {
    name: "Okafor, Amara",
    mrn: "MRN-B7K2",
    age: 62,
    gender: "F",
    complaint: "Unresponsive episode, GCS 13, post-ictal",
    esi: 1,
    waitMins: 3,
  },
  {
    name: "Reyes, Miguel",
    mrn: "MRN-C2M8",
    age: 58,
    gender: "M",
    complaint: "Acute shortness of breath, SpO2 89% on room air",
    esi: 2,
    waitMins: 5,
  },
  {
    name: "Chen, Li Wei",
    mrn: "MRN-D9P4",
    age: 34,
    gender: "F",
    complaint: "Severe abdominal pain RLQ, rebound tenderness",
    esi: 2,
    waitMins: 12,
  },
  {
    name: "Patel, Priya",
    mrn: "MRN-E4R7",
    age: 41,
    gender: "F",
    complaint: "Suspected DVT, unilateral leg swelling and pain",
    esi: 2,
    waitMins: 22,
  },
  {
    name: "Smith, Karen",
    mrn: "MRN-F6T1",
    age: 29,
    gender: "F",
    complaint: "Abdominal pain, nausea, vitals stable",
    esi: 3,
    waitMins: 40,
    alert: true,
  },
  {
    name: "Novak, Tomas",
    mrn: "MRN-G8W5",
    age: 53,
    gender: "M",
    complaint: "Kidney stone, flank pain 8/10, hematuria",
    esi: 3,
    waitMins: 47,
  },
  {
    name: "Garcia, Sofia",
    mrn: "MRN-H1Y9",
    age: 24,
    gender: "F",
    complaint: "Minor forearm laceration, bleeding controlled",
    esi: 4,
    waitMins: 15,
  },
  {
    name: "Kim, Daniel",
    mrn: "MRN-J5Q3",
    age: 38,
    gender: "M",
    complaint: "Sprained ankle, ambulatory, no deformity",
    esi: 4,
    waitMins: 31,
  },
  {
    name: "Brown, Alice",
    mrn: "MRN-K3V6",
    age: 67,
    gender: "F",
    complaint: "Prescription refill request, asymptomatic",
    esi: 5,
    waitMins: 10,
  },
  {
    name: "Ibrahim, Yusuf",
    mrn: "MRN-L8N2",
    age: 19,
    gender: "M",
    complaint: "Sore throat, low-grade fever x2 days",
    esi: 5,
    waitMins: 26,
  },
];

export const SURGE_PATIENTS: Patient[] = [
  {
    name: "MERTZ, Hanna",
    mrn: "MRN-M2X8",
    age: 71,
    gender: "F",
    complaint: "Multi-vehicle MVC, chest trauma, hypotensive",
    esi: 1,
    waitMins: 1,
  },
  {
    name: "Fischer, Otto",
    mrn: "MRN-N6C4",
    age: 55,
    gender: "M",
    complaint: "MVC walk-in, head laceration, LOC at scene",
    esi: 2,
    waitMins: 4,
  },
  {
    name: "Adeyemi, Femi",
    mrn: "MRN-P9Z1",
    age: 30,
    gender: "M",
    complaint: "MVC, wrist deformity, stable vitals",
    esi: 3,
    waitMins: 6,
  },
];

export const ESI_META: Record<
  EsiLevel,
  { label: string; badge: string; dot: string; swatch: string }
> = {
  1: {
    label: "Immediate",
    badge: "bg-esi-1 text-primary-foreground",
    dot: "bg-esi-1",
    swatch: "var(--esi-1)",
  },
  2: {
    label: "Emergent",
    badge: "bg-esi-2 text-primary-foreground",
    dot: "bg-esi-2",
    swatch: "var(--esi-2)",
  },
  3: {
    label: "Urgent",
    badge: "bg-esi-3 text-foreground",
    dot: "bg-esi-3",
    swatch: "var(--esi-3)",
  },
  4: {
    label: "Less Urgent",
    badge: "bg-esi-4 text-primary-foreground",
    dot: "bg-esi-4",
    swatch: "var(--esi-4)",
  },
  5: {
    label: "Non-Urgent",
    badge: "bg-esi-5 text-primary-foreground",
    dot: "bg-esi-5",
    swatch: "var(--esi-5)",
  },
};

export function formatWait(mins: number) {
  if (mins < 60) return `${mins} mins`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

export function checkDeterioration(patient: Patient, waitOffset: number): boolean {
  if (patient.alert) return true;
  const effectiveWait = patient.waitMins + waitOffset;
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
  addPatient: (patient: Patient) => void;
}

const TriageContext = createContext<TriageContextState | undefined>(undefined);

export function TriageProvider({ children }: { children: React.ReactNode }) {
  const [aiActive, setAiActive] = useState(true);
  const [surge, setSurge] = useState(false);
  const [waitOffset, setWaitOffset] = useState(0);
  const [sortMode, setSortMode] = useState<"wait" | "priority">("priority");
  const [searchQuery, setSearchQuery] = useState("");
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const [baseList, setBaseList] = useState<Patient[]>(() =>
    BASE_PATIENTS.map((p) => ({ ...p, addedAt: 0 })),
  );
  const [surgeList, setSurgeList] = useState<Patient[]>([]);

  // When surge is toggled, add or remove the surge patients
  useEffect(() => {
    if (surge) {
      const now = Date.now();
      setSurgeList(SURGE_PATIENTS.map((p) => ({ ...p, addedAt: now })));
    } else {
      setSurgeList([]);
    }
  }, [surge]);

  const addPatient = (patient: Patient) => {
    setBaseList((prev) => [{ ...patient, addedAt: Date.now() }, ...prev]);
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

    return filtered.sort((a, b) =>
      sortMode === "wait"
        ? b.waitMins + waitOffset - (a.waitMins + waitOffset)
        : a.esi - b.esi || b.waitMins + waitOffset - (a.waitMins + waitOffset),
    );
  }, [baseList, surgeList, sortMode, waitOffset, searchQuery]);

  const allPatientsCount = baseList.length + surgeList.length;

  return (
    <TriageContext.Provider
      value={{
        aiActive,
        setAiActive,
        surge,
        setSurge,
        waitOffset,
        setWaitOffset,
        sortMode,
        setSortMode,
        searchQuery,
        setSearchQuery,
        patients,
        allPatientsCount,
        isAddModalOpen,
        setIsAddModalOpen,
        addPatient,
      }}
    >
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
