import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowLeft, Plus } from "lucide-react";

import { Sidebar } from "../components/Sidebar";
import { Header } from "../components/Header";
import { PatientRow } from "../components/PatientRow";
import { useTriage } from "../lib/triage-context";

export const Route = createFileRoute("/patients")({
  head: () => ({
    meta: [{ title: "All Waiting Patients — TriageFlow" }],
  }),
  component: PatientsList,
});

function PatientsList() {
  const { patients, waitOffset, setWaitOffset, sortMode, setSortMode, setIsAddModalOpen } = useTriage();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background font-sans text-foreground flex">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((value) => !value)} />
      <div className={`flex-1 transition-[padding] duration-200 ${sidebarCollapsed ? "pl-20" : "pl-64"} flex flex-col min-w-0`}>
        <Header />

        <main className="space-y-6 p-6 flex-1 flex flex-col">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                to="/" 
                className="inline-flex items-center justify-center rounded-md border border-input bg-background p-2 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                aria-label="Back to Dashboard"
              >
                <ArrowLeft className="h-4 w-4" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">All Waiting Patients</h1>
                <p className="text-sm text-muted-foreground">
                  {patients.length} total patients in queue
                </p>
              </div>
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

          <section className="flex flex-col flex-1 overflow-hidden rounded-xl border border-border bg-card shadow-sm">
            <div className="grid gap-5 border-b border-border bg-muted/30 px-5 py-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
              <div className="flex min-w-0 items-center gap-4">
                <label htmlFor="wait-offset-patients" className="shrink-0 text-sm font-semibold text-foreground">Time (mins)</label>
                <input
                  id="wait-offset-patients"
                  type="range"
                  min="0"
                  max="60"
                  step="5"
                  value={waitOffset}
                  onChange={(event) => setWaitOffset(Number(event.target.value))}
                  className="h-2 min-w-0 flex-1 cursor-pointer accent-primary"
                  aria-valuetext={`${waitOffset} minutes added to wait times`}
                />
                <output htmlFor="wait-offset-patients" className="w-16 shrink-0 rounded-lg border border-border bg-card px-3 py-2 text-center text-sm font-bold tabular-nums">
                  {waitOffset}
                </output>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold text-foreground">Sort by</span>
                <div className="flex rounded-lg border border-border bg-card p-1" role="group" aria-label="Sort patient queue">
                  <button type="button" onClick={() => setSortMode("wait")} aria-pressed={sortMode === "wait"} className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${sortMode === "wait" ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
                    Wait Time
                  </button>
                  <button type="button" onClick={() => setSortMode("priority")} aria-pressed={sortMode === "priority"} className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${sortMode === "priority" ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
                    Priority
                  </button>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-[minmax(0,1.6fr)_70px_minmax(0,2.2fr)_170px_90px] gap-4 border-b border-border bg-muted/50 px-5 py-2.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
              <span>Patient / MRN</span>
              <span>Age · Sex</span>
              <span>Chief Complaint</span>
              <span>ESI Score</span>
              <span>Wait Time</span>
            </div>

            <div className="flex-1 overflow-auto">
              {patients.length > 0 ? (
                patients.map((p) => (
                  <PatientRow key={p.mrn} patient={p} />
                ))
              ) : (
                <div className="p-8 text-center text-muted-foreground text-sm">
                  No patients found.
                </div>
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
