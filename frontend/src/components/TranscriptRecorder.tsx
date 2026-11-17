/* eslint-disable prettier/prettier */
import { useRef, useState } from "react";
import { ApiError, describeError, newRequestId, transcribeAudio, verifyCapture, type CaptureOut } from "../lib/api";

/**
 * Server dictation: record (or upload) audio -> faster-whisper transcript -> nurse reviews and confirms.
 * Low-confidence and auto-corrected words are highlighted. The transcript only reaches the complaint field after
 * "Use transcript", which verifies it on the server when review is required (the backend refuses triage otherwise).
 */
export function TranscriptRecorder({ mrn, onConfirmed, onUnavailable }: {
  mrn?: string | undefined;
  onConfirmed: (text: string, captureId: string) => void;
  onUnavailable?: () => void;
}) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cap, setCap] = useState<CaptureOut | null>(null);
  const [text, setText] = useState("");
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const requestId = useRef<string>(newRequestId());

  const send = async (blob: Blob) => {
    setBusy(true); setError(null);
    try {
      const c = await transcribeAudio(blob, requestId.current, mrn);  // a retry after a network error reuses the id
      setCap(c);
      setText(c.text);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) onUnavailable?.();
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunks.current = [];
      rec.ondataavailable = (e) => { if (e.data.size) chunks.current.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        void send(new Blob(chunks.current, { type: rec.mimeType || "audio/webm" }));
      };
      rec.start();
      recorder.current = rec;
      requestId.current = newRequestId();
      setRecording(true);
    } catch {
      setError("Microphone unavailable or permission denied.");
    }
  };

  const stop = () => {
    recorder.current?.stop();
    setRecording(false);
  };

  const use = async () => {
    if (!cap) return;
    const edited = text.trim() !== cap.text.trim();
    try {
      if (cap.status === "PENDING_REVIEW" || edited) await verifyCapture(cap.id, text.trim());
      onConfirmed(text.trim(), cap.id);
      setCap(null); setText("");
    } catch (err) {
      setError(describeError(err));
    }
  };

  return (
    <div className="mt-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {!recording ? (
          <button type="button" disabled={busy} onClick={() => void start()} className="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">● Record dictation</button>
        ) : (
          <button type="button" onClick={stop} className="rounded-full bg-red-600 px-3 py-1 text-xs font-semibold text-white animate-pulse">■ Stop</button>
        )}
        <label className="cursor-pointer text-xs font-semibold text-blue-600 hover:underline">
          or upload audio
          <input type="file" accept="audio/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) { requestId.current = newRequestId(); void send(f); } }} />
        </label>
        {busy && <span className="text-xs text-muted-foreground">Transcribing…</span>}
      </div>
      {error && <p className="text-xs text-red-600">{error}</p>}
      {cap && (
        <div className="space-y-2">
          <p className="text-xs">
            <span className={`mr-2 rounded px-1.5 py-0.5 font-bold ${cap.status === "PENDING_REVIEW" ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800"}`}>
              {cap.status === "PENDING_REVIEW" ? `Review needed: ${cap.low_confidence_count} uncertain word(s)` : "High confidence"}
            </span>
            {cap.words.map((w, i) => (
              <span key={i} title={w["corrected_from"] ? `heard "${String(w["corrected_from"])}"` : `${Math.round(Number(w["probability"]) * 100)}%`}
                className={w["low"] ? (w["corrected_from"] ? "underline decoration-blue-500 decoration-2 bg-amber-100" : "bg-amber-200") : ""}>
                {String(w["word"])}{" "}
              </span>
            ))}
          </p>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} className="w-full rounded border px-2 py-1 text-sm" aria-label="Edit transcript" />
          {cap.fields.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {cap.fields.map((f, i) => (
                <span key={i} className={`rounded-full border px-2 py-0.5 text-[11px] ${f["low_confidence"] ? "border-amber-300 bg-amber-50" : "border-slate-200 bg-white"}`}>
                  {String(f["name"]).replace("_", " ")}: {String(f["value"])}
                </span>
              ))}
            </div>
          )}
          <button type="button" onClick={() => void use()} disabled={!text.trim()} className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50">
            {cap.status === "PENDING_REVIEW" ? "I checked it: use transcript" : "Use transcript"}
          </button>
        </div>
      )}
    </div>
  );
}
