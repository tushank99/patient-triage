/* eslint-disable prettier/prettier */
import { useCallback, useEffect, useRef, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import {
  DOC_TYPES, deleteDocument, describeError, fetchDocumentChunks, listDocuments, uploadDocument, uploadDocumentVersion,
  type ChunkOut, type DocumentOut,
} from "../lib/api";
import { useAuth } from "../lib/auth";
import { OcrReview } from "./OcrReview";

const STATUS_STYLE: Record<string, string> = {
  READY: "bg-emerald-50 text-emerald-700 border-emerald-200",
  PENDING: "bg-slate-50 text-slate-600 border-slate-200",
  PROCESSING: "bg-blue-50 text-blue-700 border-blue-200",
  FAILED: "bg-red-50 text-red-700 border-red-200",
  NEEDS_REVIEW: "bg-amber-50 text-amber-800 border-amber-300",
};

/** Upload, version, inspect and delete a patient's documents. Processing happens in background jobs,
 *  so the list polls while anything is pending. */
export function PatientDocumentsModal({ mrn, open, onClose }: { mrn: string; open: boolean; onClose: () => void }) {
  const { can } = useAuth();
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [docType, setDocType] = useState<string>("clinic_note");
  const [title, setTitle] = useState("");
  const [docDate, setDocDate] = useState("");
  const [chunks, setChunks] = useState<{ id: string; rows: ChunkOut[] } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const versionInput = useRef<HTMLInputElement>(null);
  const [versionTarget, setVersionTarget] = useState<string | null>(null);
  const [reviewing, setReviewing] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments(mrn));
      setError(null);
    } catch (err) {
      setError(describeError(err));
    }
  }, [mrn]);

  useEffect(() => {
    if (open) void refresh();
  }, [open, refresh]);

  const pending = docs.some((d) => d.status === "PENDING" || d.status === "PROCESSING" || d.latest_job_status === "QUEUED" || d.latest_job_status === "RUNNING");
  useEffect(() => {
    if (!open || !pending) return undefined;
    const t = setInterval(() => void refresh(), 2500);
    return () => clearInterval(t);
  }, [open, pending, refresh]);

  const doUpload = async () => {
    if (!file) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const res = await uploadDocument(mrn, file, docType, title || undefined, docDate || undefined);
      setNotice(res.status === "duplicate" ? `Already on file as "${res.document.title}" (identical file).` : "Uploaded. Processing in the background…");
      setFile(null); setTitle(""); setDocDate("");
      if (fileInput.current) fileInput.current.value = "";
      await refresh();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const doNewVersion = async (f: File) => {
    if (!versionTarget) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const res = await uploadDocumentVersion(versionTarget, f);
      setNotice(res.status === "duplicate" ? "That file is identical to the current version." : "New version uploaded; the old one stays searchable until it is processed.");
      await refresh();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false); setVersionTarget(null);
      if (versionInput.current) versionInput.current.value = "";
    }
  };

  const doDelete = async (d: DocumentOut) => {
    if (!window.confirm(`Delete "${d.title}"? It is removed from search immediately and its stored file is purged.`)) return;
    try {
      await deleteDocument(d.id);
      await refresh();
    } catch (err) {
      setError(describeError(err));
    }
  };

  const showChunks = async (d: DocumentOut) => {
    if (chunks?.id === d.id) { setChunks(null); return; }
    try {
      setChunks({ id: d.id, rows: await fetchDocumentChunks(d.id) });
    } catch (err) {
      setError(describeError(err));
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Documents · {mrn}</DialogTitle>
        </DialogHeader>
        {error && <p role="alert" className="rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{error}</p>}
        {notice && <p className="rounded border border-blue-200 bg-blue-50 p-2 text-sm text-blue-800">{notice}</p>}

        {can("document:write") && (
          <section className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Upload (PDF, DOCX, text, or a scan/photo PNG/JPEG/TIFF · max 10 MB)</h3>
            <input ref={fileInput} type="file" accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.tif,.tiff,application/pdf,text/plain,image/*,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="block w-full text-sm" />
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              <select value={docType} onChange={(e) => setDocType(e.target.value)} className="rounded-md border px-2 py-1.5 text-sm" aria-label="Document type">
                {DOC_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
              </select>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title (optional)" className="rounded-md border px-2 py-1.5 text-sm" />
              <input type="date" value={docDate} onChange={(e) => setDocDate(e.target.value)} className="rounded-md border px-2 py-1.5 text-sm" aria-label="Document date" />
            </div>
            <button type="button" disabled={!file || busy} onClick={() => void doUpload()} className="rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
              {busy ? "Uploading…" : "Upload"}
            </button>
          </section>
        )}

        <input ref={versionInput} type="file" className="hidden" accept=".pdf,.docx,.txt,.md"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) void doNewVersion(f); }} />

        <ul className="divide-y divide-slate-100">
          {docs.length === 0 && <li className="py-6 text-center text-sm text-muted-foreground">No documents on file.</li>}
          {docs.map((d) => (
            <li key={d.id} className="py-3 text-sm">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="font-semibold truncate">{d.title}</p>
                  <p className="text-xs text-muted-foreground">
                    {d.doc_type.replace(/_/g, " ")} · {d.doc_date ?? "undated"} · v{d.current_version} · {d.chunk_count} chunks
                    {d.is_synthetic ? " · synthetic" : ""}{d.uploaded_by ? ` · by ${d.uploaded_by}` : ""}
                  </p>
                  {d.latest_job_error && d.status !== "READY" && <p className="text-xs text-red-600 mt-1">{d.latest_job_error}</p>}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className={`rounded-full border px-2 py-0.5 text-[11px] font-bold ${STATUS_STYLE[d.status] ?? STATUS_STYLE["PENDING"]}`}>{d.status}</span>
                  {d.status === "NEEDS_REVIEW" && can("document:write") && (
                    <button type="button" onClick={() => setReviewing(reviewing === d.id ? null : d.id)} className="text-xs font-semibold text-amber-700 hover:underline">Review OCR</button>
                  )}
                  <button type="button" onClick={() => void showChunks(d)} className="text-xs text-blue-600 hover:underline">Chunks</button>
                  {can("document:write") && (
                    <button type="button" onClick={() => { setVersionTarget(d.id); versionInput.current?.click(); }} className="text-xs text-blue-600 hover:underline">New version</button>
                  )}
                  {can("document:delete") && (
                    <button type="button" onClick={() => void doDelete(d)} className="text-xs text-red-600 hover:underline">Delete</button>
                  )}
                </div>
              </div>
              {reviewing === d.id && <OcrReview mrn={mrn} documentId={d.id} onDone={() => { setReviewing(null); void refresh(); }} />}
              {chunks?.id === d.id && (
                <ol className="mt-2 space-y-1.5 rounded bg-slate-50 p-2">
                  {chunks.rows.map((c) => (
                    <li key={c.id} className="text-xs">
                      <span className="font-semibold">#{c.chunk_index}{c.section ? ` · ${c.section}` : ""}{c.page_start ? ` · p.${c.page_start}${c.page_end && c.page_end !== c.page_start ? `-${c.page_end}` : ""}` : ""}</span>
                      <p className="whitespace-pre-line text-slate-600">{c.text}</p>
                    </li>
                  ))}
                </ol>
              )}
            </li>
          ))}
        </ul>
      </DialogContent>
    </Dialog>
  );
}
