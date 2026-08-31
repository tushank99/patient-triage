/* eslint-disable prettier/prettier */
import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowLeft, Plus, History, List, FastForward } from "lucide-react";
import { Sidebar } from "../components/Sidebar";
import { Header } from "../components/Header";
import { PatientRow } from "../components/PatientRow";
import { ArchiveTable } from "../components/ArchiveTable";
import { useTriage } from "../lib/triage-context";

export const Route = createFileRoute("/patients")({
  head: () => ({
    meta: [{ title: "Patient Queue — TriageFlow" }],
  }),
  component: PatientsList,
});

function PatientsList() {
  const { patients, waitOffset, setWaitOffset, sortMode, setSortMode, setIsAddModalOpen } = useTriage();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  const [viewMode, setViewMode] = useState<"QUEUE" | "ARCHIVE">("QUEUE");

  return (
    <div className="min-h-screen bg-background font-sans text-foreground flex">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((value) => !value)} />
      <div className={`flex-1 transition-[padding] duration-200 ${sidebarCollapsed ? "pl-20" : "pl-64"} flex flex-col min-w-0`}>
        <Header />
        <main className="space-y-6 p-6 flex-1 flex flex-col">
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/" className="inline-flex items-center justify-center rounded-md border border-input bg-background p-2 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground" aria-label="Back to Dashboard">
                <ArrowLeft className="h-4 w-4" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">Patient Tracking</h1>
                <p className="text-sm text-muted-foreground">Manage active waitlist and view historical records.</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {/* VIEW TOGGLE */}
              <div className="flex p-1 bg-slate-100 rounded-lg mr-4">
                <button 
                  onClick={() => setViewMode("QUEUE")} 
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-md transition-all ${viewMode === "QUEUE" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                >
                  <List className="w-4 h-4" /> Active Queue
                </button>
                <button 
                  onClick={() => setViewMode("ARCHIVE")} 
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-md transition-all ${viewMode === "ARCHIVE" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
                >
                  <History className="w-4 h-4" /> Records Archive
                </button>
              </div>

              <button type="button" onClick={() => setIsAddModalOpen(true)} className="inline-flex h-10 items-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90">
                <Plus className="h-4 w-4" /> Add Patient
              </button>
            </div>
          </div>

          {viewMode === "QUEUE" ? (
            <section className="flex flex-col flex-1 overflow-hidden rounded-xl border border-border bg-card shadow-sm">
              <div className="flex flex-col lg:flex-row gap-5 border-b border-border bg-muted/30 px-5 py-4 lg:items-center lg:justify-between">
                <div className="flex min-w-0 flex-1 items-center gap-4">
                  <label htmlFor="wait-offset-patients" className="flex items-center gap-1.5 shrink-0 text-sm font-bold text-slate-700">
                    <FastForward className="w-4 h-4 text-blue-600" /> Fast-Forward Time
                  </label>
                  <input id="wait-offset-patients" type="range" min="0" max="60" step="5" value={waitOffset} onChange={(event) => setWaitOffset(Number(event.target.value))} className="h-2 min-w-0 flex-1 cursor-pointer accent-blue-600" />
                  <output htmlFor="wait-offset-patients" className="w-20 shrink-0 rounded-lg border border-blue-200 bg-blue-50 text-blue-700 px-3 py-2 text-center text-sm font-bold tabular-nums">+{waitOffset}mins</output>
                </div>
                
                {/* DIVIDER */}
                <div className="hidden lg:block w-px h-8 bg-slate-300 mx-1"></div>

                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-sm font-semibold text-foreground">Sort by</span>
                  <div className="flex rounded-lg border border-border bg-card p-1" role="group">
                    <button type="button" onClick={() => setSortMode("wait")} aria-pressed={sortMode === "wait"} className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${sortMode === "wait" ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>Wait Time</button>
                    <button type="button" onClick={() => setSortMode("priority")} aria-pressed={sortMode === "priority"} className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${sortMode === "priority" ? "bg-accent text-accent-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>Priority</button>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-[minmax(0,1.6fr)_70px_minmax(0,2.2fr)_170px_140px] gap-4 border-b border-border bg-muted/50 px-5 py-2.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
                <span>Patient / MRN</span><span>Age · Sex</span><span>Chief Complaint</span><span>ESI Score</span><span className="text-right pr-4">Wait Time</span>
              </div>
              <div className="flex-1 overflow-auto">
                {patients.length > 0 ? (
                  patients.map((p) => <PatientRow key={p.mrn} patient={p} />)
                ) : (
                  <div className="p-8 text-center text-muted-foreground text-sm">No patients found.</div>
                )}
              </div>
            </section>
          ) : (
            <ArchiveTable />
          )}

        </main>
      </div>
    </div>
  );
}