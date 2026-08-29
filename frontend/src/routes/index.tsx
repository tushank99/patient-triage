import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Plus,
  Timer,
  Users,
} from "lucide-react";

import { Sidebar } from "../components/Sidebar";
import { Header } from "../components/Header";
import { PatientRow } from "../components/PatientRow";
import { EsiLevel, Patient, ESI_META, formatWait, useTriage } from "../lib/triage-context";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TriageFlow — Emergency Department Triage Dashboard" },
      {
        name: "description",
        content:
          "Real-time ED triage nurse dashboard: ESI-prioritized patient queue, deterioration alerts, and live census analytics.",
      },
      { property: "og:title", content: "TriageFlow — ED Triage Nurse Dashboard" },
      {
        property: "og:description",
        content:
          "ESI-prioritized patient queue with AI triage assistance and deterioration monitoring for emergency departments.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: TriageDashboard,
});

/* ---------- Stats ---------- */

function StatsRow({ patients, alerts, waitOffset }: { patients: Patient[]; alerts: number; waitOffset: number }) {
  const longest = patients.length > 0 ? Math.max(...patients.map((p) => p.waitMins + waitOffset)) : 0;
  const stats = [
    { label: "Total Patients", value: String(patients.length), icon: Users, note: "+6 in last hour" },
    { label: "Active Staff", value: "8", icon: Activity, note: "3 MD · 5 RN" },
    { label: "Longest Wait", value: formatWait(longest).replace("mins", "m"), icon: Timer, note: "ESI 3 threshold 60m" },
    { label: "Triage Alerts", value: String(alerts), icon: AlertTriangle, note: "Needs reassessment" },
  ];
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {stats.map(({ label, value, icon: Icon, note }) => (
        <div key={label} className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <Icon className="h-4 w-4 text-muted-foreground" />
          </div>
          <p className="mt-2 text-3xl font-bold tracking-tight">{value}</p>
          <p className="mt-1 text-xs text-muted-foreground">{note}</p>
        </div>
      ))}
    </div>
  );
}

/* ---------- ESI donut ---------- */

function EsiDonut({ patients }: { patients: Patient[] }) {
  const counts = useMemo(() => {
    const c: Record<EsiLevel, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    for (const p of patients) c[p.esi] += 1;
    return c;
  }, [patients]);
  const total = patients.length;

  const R = 70;
  const C = 2 * Math.PI * R;
  let offset = 0;
  const segments = ([1, 2, 3, 4, 5] as EsiLevel[])
    .filter((l) => counts[l] > 0)
    .map((l) => {
      const frac = counts[l] / total;
      const seg = { level: l, dash: frac * C, offset };
      offset += frac * C;
      return seg;
    });

  return (
    <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
      <h3 className="text-base font-bold tracking-tight">Current ESI Distribution</h3>
      <p className="text-xs text-muted-foreground">Live census by acuity level</p>
      <div className="mt-4 flex items-center gap-6">
        <div className="relative h-44 w-44 shrink-0">
          <svg viewBox="0 0 180 180" className="h-full w-full -rotate-90">
            <circle cx="90" cy="90" r={R} fill="none" stroke="var(--muted)" strokeWidth="20" />
            {segments.map((s) => (
              <circle
                key={s.level}
                cx="90"
                cy="90"
                r={R}
                fill="none"
                stroke={ESI_META[s.level].swatch}
                strokeWidth="20"
                strokeDasharray={`${s.dash} ${C - s.dash}`}
                strokeDashoffset={-s.offset}
                strokeLinecap="butt"
              />
            ))}
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-bold">{total}</span>
            <span className="text-xs text-muted-foreground">Patients</span>
          </div>
        </div>
        <ul className="flex-1 space-y-2">
          {([1, 2, 3, 4, 5] as EsiLevel[]).map((l) => (
            <li key={l} className="flex items-center gap-2.5 text-sm">
              <span className={`h-2.5 w-2.5 rounded-full ${ESI_META[l].dot}`} />
              <span className="flex-1 font-medium text-foreground">
                ESI {l} <span className="font-normal text-muted-foreground">· {ESI_META[l].label}</span>
              </span>
              <span className="font-bold tabular-nums">{counts[l]}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/* ---------- Page ---------- */

function TriageDashboard() {
  const { patients, surge, waitOffset, setIsAddModalOpen } = useTriage();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // We consider a patient alerting if waitMins + offset triggers the condition.
  // Wait, the stats row was just filtering p.alert. For stats, let's keep it simple:
  // we count how many are deteriorated based on the same logic used in PatientRow.
  const alerts = patients.filter((p) => {
    if (p.alert) return true;
    const effectiveWait = p.waitMins + waitOffset;
    if (p.esi === 2 && effectiveWait >= 15) return true;
    if (p.esi === 3 && effectiveWait >= 60) return true;
    return false;
  }).length;

  const topPatients = patients.slice(0, 6);

  return (
    <div className="min-h-screen bg-background font-sans text-foreground">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((value) => !value)} />
      <div className={`transition-[padding] duration-200 ${sidebarCollapsed ? "pl-20" : "pl-64"}`}>
        <Header />

        <main className="space-y-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Triage Dashboard</h1>
              <p className="text-sm text-muted-foreground">
                {surge
                  ? "Mass-casualty surge protocol active — incoming MVC patients."
                  : "Live emergency department queue · Unit A"}
              </p>
            </div>
            {alerts > 0 && (
              <span className="inline-flex items-center gap-1.5 rounded-lg border border-alert-border bg-alert-bg px-3 py-2 text-xs font-bold text-esi-1">
                <AlertTriangle className="h-3.5 w-3.5" />
                {alerts} deterioration alert{alerts > 1 ? "s" : ""}
              </span>
            )}
          </div>

           <StatsRow patients={patients} alerts={alerts} waitOffset={waitOffset} />

          <section className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
             <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border px-5 py-4">
               <div>
                 <h2 className="text-lg font-bold tracking-tight">Patient Queue</h2>
                 <p className="text-xs text-muted-foreground">Showing top 6 of {patients.length} patients · live triage order</p>
               </div>
               <button 
                 type="button" 
                 onClick={() => setIsAddModalOpen(true)}
                 className="inline-flex h-10 items-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90"
               >
                <Plus className="h-4 w-4" />
                Add Patient
              </button>
            </div>

            <div className="grid grid-cols-[minmax(0,1.6fr)_70px_minmax(0,2.2fr)_170px_90px] gap-4 border-b border-border bg-muted/50 px-5 py-2.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
              <span>Patient / MRN</span>
              <span>Age · Sex</span>
              <span>Chief Complaint</span>
              <span>ESI Score</span>
              <span>Wait Time</span>
            </div>

            <div>
              {topPatients.map((p) => (
                 <PatientRow key={p.mrn} patient={p} />
              ))}
            </div>
            
            {patients.length > 6 && (
              <div className="border-t border-border bg-muted/10 p-3 text-center">
                <Link
                  to="/patients"
                  className="inline-block w-full rounded-md py-2.5 text-sm font-semibold text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  Show All Patients
                </Link>
              </div>
            )}
          </section>

          <div className="flex justify-end">
            <div className="w-full max-w-md">
              <EsiDonut patients={patients} />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
