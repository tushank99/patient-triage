"""
Speech-to-text benchmark: clinical complaints synthesised with the OS text-to-speech voices (Windows SAPI), clean and
with added noise, transcribed by faster-whisper models with and without the medical vocabulary (prompt + correction).

  python -m backend.eval.stt_benchmark --models tiny.en,base.en --report docs/STT_RESULTS.md

Metrics: word error rate (WER), medical-term recall (drug/condition words in the reference recognised exactly),
extraction accuracy (symptoms F1, duration, pain score from the transcript vs gold), the review gate's sentence-level
error catch rate (sentences with any error that were flagged for review) and false-review rate, and real-time factor.
TTS speech is far cleaner than a noisy ED: this compares configurations; it does not certify real-world accuracy.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.intake import extract_speech  # noqa: E402
from backend.app.intake.stt import FasterWhisperTranscriber, vocabulary, wer  # noqa: E402

# (text, gold symptoms, gold duration, gold pain score)
SENTENCES = [
    ("I have had crushing chest pain for two hours and I take apixaban for atrial fibrillation.", {"chest pain"}, "2 hours", None),
    ("She fell at home this morning and hit her head, she is on warfarin.", {"fall", "head injury"}, "since this morning", None),
    ("Short of breath for three days, my inhaler with albuterol is not helping.", {"shortness of breath"}, "3 days", None),
    ("Pain is eight out of ten in my lower back since yesterday.", {"back pain"}, "since yesterday", 8),
    ("He has a fever and a cough for five days, no chest pain.", {"fever", "cough"}, "5 days", None),
    ("Sudden headache, the worst headache of my life, started one hour ago.", {"headache"}, "1 hours", None),
    ("I am allergic to penicillin and I took amoxicillin yesterday, now I have a rash.", {"rash"}, None, None),
    ("Dizzy and nauseous for two days, I take metoprolol and lisinopril.", {"dizziness", "nausea"}, "2 days", None),
    ("My heart is racing, palpitations for thirty minutes, I take levothyroxine.", {"palpitations"}, "30 minutes", None),
    ("Burning when I pee for three days, pain four out of ten.", {"urinary symptoms"}, "3 days", 4),
    ("Weakness in my left arm and slurred speech since this morning.", {"weakness", "slurred speech"}, "since this morning", None),
    ("Leg swelling for two weeks, I take furosemide and metformin.", {"leg swelling"}, "2 weeks", None),
    ("Vomiting since last night with abdominal pain, seven out of ten.", {"vomiting", "abdominal pain"}, "since last night", 7),
    ("He had a seizure, he takes levetiracetam, he is confused now.", {"seizure", "confusion"}, None, None),
    ("Sore throat for four days, no fever, I have no allergies.", {"sore throat"}, "4 days", None),
    ("Chest tightness and shortness of breath for one hour, I use rivaroxaban.", {"chest pain", "shortness of breath"}, "1 hours", None),
]


def synthesize(text: str, voice: str, path: str) -> None:
    ps = (f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.Rate = 0; $s.SetOutputToWaveFile('{path}'); $s.Speak(@'\n{text}\n'@); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True, timeout=60)


def voices() -> list:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | % { $_.VoiceInfo.Name }"],
                         capture_output=True, text=True, timeout=60).stdout.split("\n")
    return [v.strip() for v in out if v.strip()]


def add_noise(wav: bytes, snr_db: float, seed: int) -> bytes:
    with wave.open(io.BytesIO(wav)) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0, 1, x.shape)
    noise *= np.sqrt(np.mean(x ** 2) / (10 ** (snr_db / 10))) / (np.std(noise) or 1)
    y = np.clip(x + noise, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setparams(params)
        o.writeframes(y.tobytes())
    return buf.getvalue()


def term_recall(ref: str, hyp: str) -> tuple[int, int]:
    vocab = {w for v in vocabulary() for w in v.lower().split() if len(w) >= 5}
    ref_terms = [w.strip(".,") for w in ref.lower().split() if w.strip(".,") in vocab]
    hyp_words = {w.strip(".,") for w in hyp.lower().split()}
    return sum(t in hyp_words for t in ref_terms), len(ref_terms)


def extraction_scores(text: str, gold_sym, gold_dur, gold_pain):
    fields = extract_speech.extract(text)
    syms = {f.value for f in fields if f.name == "symptom"}
    tp = len(syms & gold_sym)
    p = tp / len(syms) if syms else (1.0 if not gold_sym else 0.0)
    r = tp / len(gold_sym) if gold_sym else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    durs = [f.value for f in fields if f.name == "duration"]
    pain = next((int(f.value) for f in fields if f.name == "pain_score"), None)
    return f1, (gold_dur in durs) if gold_dur else (not durs), pain == gold_pain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="tiny.en,base.en")
    ap.add_argument("--out", default=str(ROOT / "backend" / "eval" / "stt_results.json"))
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="stt-bench-")
    vs = voices()
    clips = []
    for i, (text, *_rest) in enumerate(SENTENCES):
        path = os.path.join(tmp, f"{i}.wav")
        synthesize(text, vs[i % len(vs)], path)
        clean = Path(path).read_bytes()
        clips.append((i, "clean", clean))
        clips.append((i, "noise 10 dB SNR", add_noise(clean, 10, i)))
    results = {}
    for model in [m for m in a.models.split(",") if m]:
        tr = FasterWhisperTranscriber(model, "int8")
        for vocab_on in (False, True):
            for cond in ("clean", "noise 10 dB SNR"):
                rows = []
                for i, c, audio in clips:
                    if c != cond:
                        continue
                    ref, gs, gd, gp = SENTENCES[i]
                    t0 = time.perf_counter()
                    t = tr.transcribe(audio, "en", prompt=vocab_on, correct=vocab_on)
                    secs = time.perf_counter() - t0
                    w = wer(ref, t.text)
                    hit, tot = term_recall(ref, t.text)
                    f1, dur_ok, pain_ok = extraction_scores(t.text, gs, gd, gp)
                    rows.append({"wer": w["wer"], "errors": w["errors"], "hit": hit, "tot": tot, "f1": f1, "dur": dur_ok, "pain": pain_ok,
                                 "flagged": bool(t.low_confidence_words or t.corrections), "rtf": secs / (t.duration_s or 1), "hyp": t.text})
                key = f"{model} | {'medical vocab' if vocab_on else 'plain'} | {cond}"
                err = [r for r in rows if r["errors"] > 0]
                ok = [r for r in rows if r["errors"] == 0]
                results[key] = {
                    "wer": sum(r["errors"] for r in rows) / sum(len(SENTENCES[i][0].split()) for i, c, _ in clips if c == cond),
                    "medical_term_recall": sum(r["hit"] for r in rows) / max(1, sum(r["tot"] for r in rows)),
                    "symptom_f1": float(np.mean([r["f1"] for r in rows])), "duration_acc": float(np.mean([r["dur"] for r in rows])),
                    "pain_acc": float(np.mean([r["pain"] for r in rows])),
                    "error_catch_rate": (sum(r["flagged"] for r in err) / len(err)) if err else None,
                    "false_review_rate": (sum(r["flagged"] for r in ok) / len(ok)) if ok else None,
                    "rtf": float(np.mean([r["rtf"] for r in rows])), "examples": [r["hyp"] for r in rows[:3]],
                }
                r = results[key]
                print(f"{key:<46} WER {r['wer']:.3f}  med-term recall {r['medical_term_recall']:.2f}  symptoms F1 {r['symptom_f1']:.2f}  "
                      f"errors flagged {r['error_catch_rate']}  RTF {r['rtf']:.2f}")
    Path(a.out).write_text(json.dumps({"voices": vs, "n_sentences": len(SENTENCES), "results": results}, indent=2), encoding="utf-8")
    if a.report:
        f = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        lines = ["# Speech-to-text benchmark (faster-whisper, int8 CPU)", "",
                 f"{len(SENTENCES)} clinical complaints x {len(vs)} Windows TTS voices, clean and with white noise at 10 dB SNR.", "",
                 "| model / vocabulary / audio | WER | medical-term recall | symptom F1 | duration | pain score | errors flagged for review | error-free but flagged | real-time factor |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for k, r in results.items():
            lines.append(f"| {k} | {r['wer']:.3f} | {r['medical_term_recall']:.2f} | {r['symptom_f1']:.2f} | {r['duration_acc']:.2f} | {r['pain_acc']:.2f} | "
                         f"{f(r['error_catch_rate'])} | {f(r['false_review_rate'])} | {r['rtf']:.2f} |")
        Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
