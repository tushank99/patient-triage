/* eslint-disable prettier/prettier */
/* eslint-disable react-refresh/only-export-components */
/**
 * Client state for the triage board. The BACKEND is the source of truth: the queue, the archive,
 * overrides, dispositions and vitals history all live in the database and are fetched through api.ts.
 * This context only holds view state (sort, search, modals, a "fast-forward" display offset) and a
 * polled copy of what the server says.
 */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  acknowledgeReview as apiAck,
  anonymizePatient as apiAnonymize,
  describeError,
  dispositionVisit,
  endSurge,
  fetchArchive,
  fetchQueue,
  revertDisposition,
  startSurge,
  type ArchiveItem,
  type Disposition,
  type QueueItem,
} from "./api";
import { useAuth } from "./auth";

export type EsiLevel = 1 | 2 | 3 | 4 | 5;

export interface Patient {
  /** The backend visit id. */
  id: string;
  name: string;
  mrn: string;
  age: number;
  gender: "M" | "F" | "O";
  complaint: string;
  esi: EsiLevel;
  aiEsi?: EsiLevel | undefined;
  waitMins: number;
  /** Server flagged the visit as past its safe wait (status NEEDS_REASSESSMENT). */
  alert?: boolean;
  addedAt?: number;
  /** ML model probability as a percentage. Not a blended confidence: see ConfidenceBreakdown. */
  aiConfidence?: number | undefined;
  needsReview: boolean;
  reviewReasons: string[];
  isOverridden: boolean;
  isDemoSurge: boolean;
}

export interface ArchivedPatient extends Patient {
  disposition: string;
  dispositionTime: string;
  isAnonymized: boolean;
}

export const DISPOSITION_API: Record<string, Disposition> = {
  "Admitted to ER Bed": "ADMITTED",
  "Discharged / Re-routed": "DISCHARGED",
  LWBS: "LWBS",
};
export const DISPOSITION_LABEL: Record<string, string> = {
  ADMITTED: "Admitted to ER Bed",
  DISCHARGED: "Discharged / Re-routed",
  LWBS: "LWBS",
};

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
  if (patient.esi === 1 && effectiveWait >= 5) return true;
  if (patient.esi === 2 && effectiveWait >= 15) return true;
  if (patient.esi === 3 && effectiveWait >= 60) return true;
  return false;
}

const clampEsi = (n: number | null | undefined): EsiLevel => (Math.min(5, Math.max(1, Math.round(n ?? 3))) as EsiLevel);

function fromQueue(q: QueueItem): Patient {
  return {
    id: q.visit_id,
    name: q.patient_name ?? "Patient (name hidden)",
    mrn: q.patient_mrn,
    age: Math.floor(q.age),
    gender: q.gender,
    complaint: q.complaint,
    esi: clampEsi(q.esi_score),
    aiEsi: q.ai_esi_score == null ? undefined : clampEsi(q.ai_esi_score),
    waitMins: Math.max(0, Math.floor(q.wait_time_minutes)),
    alert: q.status === "NEEDS_REASSESSMENT",
    addedAt: new Date(q.arrival_time).getTime(),
    aiConfidence: q.ai_model_probability == null ? undefined : Math.round(q.ai_model_probability * 100),
    needsReview: q.needs_review,
    reviewReasons: q.review_reasons,
    isOverridden: q.is_overridden,
    isDemoSurge: q.is_demo_surge,
  };
}

function fromArchive(a: ArchiveItem): ArchivedPatient {
  return {
    id: a.visit_id,
    name: a.patient_name ?? "—",
    mrn: a.patient_mrn,
    age: Math.floor(a.age),
    gender: a.gender,
    complaint: a.complaint,
    esi: clampEsi(a.esi_score),
    aiEsi: a.ai_esi_score == null ? undefined : clampEsi(a.ai_esi_score),
    waitMins: 0,
    addedAt: new Date(a.arrival_time).getTime(),
    aiConfidence: a.ai_model_probability == null ? undefined : Math.round(a.ai_model_probability * 100),
    needsReview: false,
    reviewReasons: [],
    isOverridden: a.is_overridden,
    isDemoSurge: false,
    disposition: DISPOSITION_LABEL[a.disposition] ?? a.disposition,
    dispositionTime: a.disposition_time ?? new Date().toISOString(),
    isAnonymized: a.is_anonymized,
  };
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
  isLoading: boolean;
  loadError: string | null;
  /** Re-fetch the queue (and archive) from the server. */
  refresh: () => Promise<void>;
  isAddModalOpen: boolean;
  setIsAddModalOpen: (val: boolean) => void;
  updateVitalsModal: { isOpen: boolean; patientId: string | null };
  setUpdateVitalsModal: (state: { isOpen: boolean; patientId: string | null }) => void;
  dispositionModal: { isOpen: boolean; patientId: string | null };
  setDispositionModal: (state: { isOpen: boolean; patientId: string | null }) => void;
  /** Disposition a visit. `disposition` is the UI label; `log.checks` is the LWBS checklist. */
  removePatient: (id: string, disposition: string, log: Record<string, unknown>) => Promise<void>;
  archivedPatients: ArchivedPatient[];
  revertPatient: (id: string) => Promise<void>;
  anonymizePatient: (id: string) => Promise<void>;
  acknowledgeReview: (id: string) => Promise<void>;
}

const TriageContext = createContext<TriageContextState | undefined>(undefined);
const POLL_MS = 8000;

export function TriageProvider({ children }: { children: React.ReactNode }) {
  const { user, can } = useAuth();
  const [aiActive, setAiActive] = useState(true);
  const [waitOffset, setWaitOffset] = useState(0);
  const [sortMode, setSortMode] = useState<"wait" | "priority">("priority");
  const [searchQuery, setSearchQuery] = useState("");
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [updateVitalsModal, setUpdateVitalsModal] = useState<{ isOpen: boolean; patientId: string | null }>({ isOpen: false, patientId: null });
  const [dispositionModal, setDispositionModal] = useState<{ isOpen: boolean; patientId: string | null }>({ isOpen: false, patientId: null });

  const [queue, setQueue] = useState<Patient[]>([]);
  const [archive, setArchive] = useState<ArchivedPatient[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const canArchive = can("archive:read");
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (!user || inFlight.current) return;
    inFlight.current = true;
    try {
      const [q, a] = await Promise.all([fetchQueue(), canArchive ? fetchArchive() : Promise.resolve([])]);
      setQueue(q.map(fromQueue));
      setArchive(a.map(fromArchive));
      setLoadError(null);
    } catch (err) {
      setLoadError(describeError(err));
    } finally {
      inFlight.current = false;
      setIsLoading(false);
    }
  }, [user, canArchive]);

  useEffect(() => {
    if (!user) return undefined;
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [user, refresh]);

  const surge = useMemo(() => queue.some((p) => p.isDemoSurge), [queue]);

  const setSurge = (on: boolean) => {
    void (async () => {
      try {
        await (on ? startSurge() : endSurge());
        await refresh();
      } catch (err) {
        alert(describeError(err));
      }
    })();
  };

  const removePatient = async (id: string, disposition: string, log: Record<string, unknown>) => {
    const api = DISPOSITION_API[disposition];
    if (!api) throw new Error(`Unknown disposition: ${disposition}`);
    const checks = (log["checks"] ?? undefined) as Record<string, boolean> | undefined;
    try {
      await dispositionVisit(id, api, checks);
      await refresh();
    } catch (err) {
      alert(`Could not save disposition: ${describeError(err)}`);
    }
  };

  const revertPatient = async (id: string) => {
    try {
      await revertDisposition(id);
      await refresh();
    } catch (err) {
      alert(describeError(err));
    }
  };

  const anonymizePatient = async (id: string) => {
    const mrn = archive.find((p) => p.id === id)?.mrn;
    if (!mrn) return;
    try {
      await apiAnonymize(mrn);
      await refresh();
    } catch (err) {
      alert(describeError(err));
    }
  };

  const acknowledgeReview = async (id: string) => {
    try {
      await apiAck(id);
      await refresh();
    } catch (err) {
      alert(describeError(err));
    }
  };

  const patients = useMemo(() => {
    const q = searchQuery.toLowerCase();
    const filtered = queue.filter(
      (p) => !q || p.name.toLowerCase().includes(q) || p.mrn.toLowerCase().includes(q) || p.complaint.toLowerCase().includes(q),
    );
    return [...filtered].sort((a, b) =>
      sortMode === "wait"
        ? b.waitMins - a.waitMins
        : a.esi - b.esi || b.waitMins - a.waitMins,
    );
  }, [queue, sortMode, searchQuery]);

  return (
    <TriageContext.Provider value={{
      aiActive, setAiActive, surge, setSurge, waitOffset, setWaitOffset, sortMode, setSortMode,
      searchQuery, setSearchQuery, patients, allPatientsCount: queue.length, isLoading, loadError, refresh,
      isAddModalOpen, setIsAddModalOpen, updateVitalsModal, setUpdateVitalsModal, dispositionModal, setDispositionModal,
      removePatient, archivedPatients: archive, revertPatient, anonymizePatient, acknowledgeReview,
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
