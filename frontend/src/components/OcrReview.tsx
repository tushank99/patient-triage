/* eslint-disable prettier/prettier */
import { useEffect, useState } from "react";
import { describeError, listCaptures, rejectCapture, verifyCapture, type CaptureOut } from "../lib/api";

/** Review a scanned document held for verification: low-confidence lines highlighted, extracted fields with
 *  confidence, corrected text editable. Verify indexes the corrected text; Reject marks the document failed. */
export function OcrReview({ mrn, documentId, onDone }: { mrn: string; documentId: string; onDone: () => void }) {
  const [cap, setCap] = useState<CaptureOut | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listCaptures(mrn, "PENDING_REVIEW")
      .then((caps) => {
        const c = caps.find((x) => x.document_id === documentId) ?? null;
        setCap(c);
        setText(c?.text ?? "");
      })
      .catch((err) => setError(describeError(err)));
  }, [mrn, documentId]);

  if (error) return <p className="text-xs text-red-600">{error}</p>;
  if (!cap) return <p className="text-xs text-muted-foreground">Loading OCR output…</p>;

  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      onDone();
    } catch (err) {
      setError(describeError(err));
    }
  };

  return (
    <div className="mt-2 space-y-2 rounded border border-amber-200 bg-amber-50 p-3 text-xs">
      <p className="font-semibold">OCR ({cap.engine}) · mean confidence {cap.confidence != null ? Math.round(cap.confidence * 100) : "?"}% · {cap.low_confidence_count} uncertain line(s). Not searchable until verified.</p>
      <ol className="max-h-40 overflow-y-auto rounded bg-white p-2">
        {cap.lines.map((l, i) => (
          <li key={i} className={l["low"] ? "bg-amber-200" : ""}>{String(l["text"])} <span className="text-muted-foreground">({Math.round(Number(l["confidence"]) * 100)}%)</span></li>
        ))}
      </ol>
      {cap.fields.length > 0 && (
        <table className="w-full bg-white">
          <thead><tr className="text-left"><th>Field</th><th>Value</th><th>Normalised</th><th>Conf.</th></tr></thead>
          <tbody>
            {cap.fields.map((f, i) => (
              <tr key={i} className={f["low_confidence"] ? "bg-amber-100" : ""}>
                <td>{String(f["kind"])}: {String(f["name"])}</td>
                <td>{String(f["value"] ?? "")} {String(f["unit"] ?? "")}</td>
                <td>{f["normalized_value"] != null ? `${String(f["normalized_value"])} ${String(f["normalized_unit"] ?? "")}` : ""} {f["note"] ? <em className="text-muted-foreground">({String(f["note"])})</em> : null}</td>
                <td>{Math.round(Number(f["confidence"]) * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} className="w-full rounded border bg-white px-2 py-1" aria-label="Corrected text" />
      <div className="flex gap-2">
        <button type="button" onClick={() => void act(() => verifyCapture(cap.id, text))} className="rounded bg-emerald-600 px-3 py-1 font-semibold text-white">Verify and index</button>
        <button type="button" onClick={() => { const r = window.prompt("Reason for rejecting this scan?"); if (r) void act(() => rejectCapture(cap.id, r)); }} className="rounded bg-slate-200 px-3 py-1 font-semibold">Reject</button>
      </div>
    </div>
  );
}
