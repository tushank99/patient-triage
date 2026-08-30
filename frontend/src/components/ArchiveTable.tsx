/* eslint-disable prettier/prettier */
import React, { useState, useEffect } from "react";
import { useTriage, ESI_META, ArchivedPatient } from "../lib/triage-context";
import { Clock, ShieldAlert, Undo2, UserX } from "lucide-react";
import { HistoricalAuditModal } from "./HistoricalAuditModal";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";

export function ArchiveTable() {
  const { archivedPatients, revertPatient, anonymizePatient } = useTriage();
  const [searchTerm, setSearchTerm] = useState("");
  const [now, setNow] = useState(Date.now());
  
  // State for the modal
  const [selectedPatient, setSelectedPatient] = useState<ArchivedPatient | null>(null);
  const [patientToPurge, setPatientToPurge] = useState<string | null>(null);

  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 30000);
    return () => clearInterval(interval);
  }, []);

  const filteredArchive = archivedPatients.filter((p) => {
    const isRecent = (now - new Date(p.dispositionTime).getTime()) <= 7 * 24 * 60 * 60 * 1000;
    const matchesSearch =
      p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.mrn.toLowerCase().includes(searchTerm.toLowerCase());
    return isRecent && matchesSearch;
  });

  return (
    <div className="p-6 bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Patient Records Archive</h2>
          <p className="text-sm text-slate-500">
            Showing records from the last 7 days to ensure optimal database performance.
          </p>
        </div>
        <input
          type="text"
          placeholder="Search MRN or Name..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-64 px-4 py-2 text-sm border rounded-lg outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200 text-sm font-semibold text-slate-500 uppercase tracking-wider">
              <th className="py-3 px-4">Patient</th>
              <th className="py-3 px-4">Final ESI</th>
              <th className="py-3 px-4">Disposition</th>
              <th className="py-3 px-4">Timestamp</th>
              <th className="py-3 px-4 text-right">Compliance Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredArchive.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-8 text-center text-slate-500">
                  No archived records found.
                </td>
              </tr>
            ) : (
              filteredArchive.map((patient) => {
                const meta = ESI_META[patient.esi];
                const dispTime = new Date(patient.dispositionTime).getTime();
                const minutesAgo = (now - dispTime) / 60000;
                const canRevert = minutesAgo <= 15;

                return (
                  <tr 
                    key={patient.id} 
                    onClick={() => setSelectedPatient(patient)}
                    className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer"
                  >
                    <td className="py-3 px-4">
                      <p className={`font-semibold ${patient.isAnonymized ? 'text-slate-400 italic' : 'text-slate-900'}`}>
                        {patient.name}
                      </p>
                      <p className="text-xs text-slate-500">{patient.mrn}</p>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-bold ${meta.badge}`}>
                        ESI {patient.esi}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-medium text-sm text-slate-700">
                      {patient.disposition}
                    </td>
                    <td className="py-3 px-4 text-sm text-slate-500 flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      {new Date(patient.dispositionTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex justify-end gap-2">
                        {canRevert && !patient.isAnonymized ? (
                          <button
                            onClick={(e) => { e.stopPropagation(); revertPatient(patient.id); }}
                            className="flex items-center gap-1 px-3 py-1 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 text-xs font-bold transition-colors"
                          >
                            <Undo2 className="w-3 h-3" /> Revert ({(15 - minutesAgo).toFixed(0)}m left)
                          </button>
                        ) : (
                          <span className="flex items-center gap-1 px-3 py-1 bg-slate-100 text-slate-400 rounded text-xs font-bold cursor-not-allowed">
                            <ShieldAlert className="w-3 h-3" /> Immutable
                          </span>
                        )}

                        {!patient.isAnonymized && (
                          <button
                            onClick={(e) => { e.stopPropagation(); setPatientToPurge(patient.id); }}
                            className="flex items-center gap-1 px-3 py-1 bg-red-50 text-red-600 border border-red-200 rounded hover:bg-red-100 text-xs font-bold transition-colors"
                          >
                            <UserX className="w-3 h-3" /> Purge PII
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <HistoricalAuditModal 
        patient={selectedPatient}
        isOpen={!!selectedPatient}
        onClose={() => setSelectedPatient(null)}
      />

      {/* Custom Purge Confirmation Modal */}
      <Dialog open={!!patientToPurge} onOpenChange={(open) => !open && setPatientToPurge(null)}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="text-red-600 flex items-center gap-2">
              <ShieldAlert className="w-5 h-5" />
              Confirm PII Purge
            </DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <p className="text-sm text-slate-600">
              <strong>WARNING:</strong> This action will irreversibly redact the patient's Name and MRN from this record to comply with GDPR/HIPAA regulations.
            </p>
            <p className="text-sm text-slate-600 mt-2">
              Clinical data (vitals, ESI score, and Rationale) will be permanently retained for anonymized ML training.
            </p>
          </div>
          <div className="flex justify-end gap-3 mt-2">
            <button
              onClick={() => setPatientToPurge(null)}
              className="px-4 py-2 text-sm font-medium text-slate-600 bg-slate-100 rounded-md hover:bg-slate-200 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                if (patientToPurge) {
                  anonymizePatient(patientToPurge);
                  setPatientToPurge(null);
                }
              }}
              className="px-4 py-2 text-sm font-bold text-white bg-red-600 rounded-md hover:bg-red-700 transition-colors"
            >
              Confirm Purge
            </button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}