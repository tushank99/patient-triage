"""
Speech-to-text for intake (faster-whisper, CTranslate2 int8 on CPU; model loads lazily on first use).

  * Medical vocabulary: the decoder's initial_prompt carries drug and condition names (from the drug/lexicon files and
    MEDICAL_TERMS), biasing recognition toward them. After decoding, LOW-CONFIDENCE words are fuzzy-matched against the
    vocabulary ("a pixaban" -> "apixaban"); every substitution is listed, never silent.
  * Confidence: per-word probability from the decoder; words below STT_MIN_WORD_CONFIDENCE are flagged for review.
  * Robustness: Silero VAD drops silence/noise; temperature fallback retries hard segments; audio is validated
    (magic bytes, size, duration) before decoding. Upload retries are idempotent (request_id).
  * Language: STT_LANGUAGE (ISO code) or "auto" to detect.
The transcript is a DRAFT: it becomes clinical text only when a clinician confirms it (captures.verify).
"""
import difflib
import io
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from backend.app.clinical.facts import drug_data
from backend.app.config import settings
from backend.app.retrieval.normalize import lexicon

logger = logging.getLogger(__name__)

MEDICAL_TERMS = [
    "apixaban", "warfarin", "rivaroxaban", "metoprolol", "lisinopril", "losartan", "amlodipine", "metformin", "insulin",
    "levothyroxine", "furosemide", "albuterol", "levetiracetam", "atorvastatin", "aspirin", "clopidogrel", "prednisone",
    "amoxicillin", "penicillin", "ibuprofen", "acetaminophen", "nitroglycerin", "sumatriptan", "gabapentin", "tiotropium",
    "atrial fibrillation", "hypertension", "diabetes", "asthma", "COPD", "anaphylaxis", "syncope", "dyspnea", "palpitations",
    "myocardial infarction", "stroke", "seizure", "epilepsy", "pneumonia", "anticoagulant", "migraine", "nausea", "vomiting",
]


class AudioRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def sniff_audio(data: bytes) -> Optional[str]:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm"  # MediaRecorder default in Chrome/Firefox
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    return None


def validate_audio(data: bytes) -> str:
    if not data:
        raise AudioRejected(400, "Empty audio file.")
    if len(data) > settings.max_audio_mb * 1024 * 1024:
        raise AudioRejected(413, f"Audio is larger than {settings.max_audio_mb:g} MB.")
    mime = sniff_audio(data)
    if mime is None:
        raise AudioRejected(415, "Unsupported audio. Accepted: WAV, WebM, Ogg, MP3, M4A/MP4, FLAC.")
    try:
        import av

        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioRejected(415, "File contains no audio stream.")
            seconds = float(container.duration / 1_000_000) if container.duration else None
    except AudioRejected:
        raise
    except Exception:
        raise AudioRejected(415, "Audio could not be decoded (corrupt or unsupported codec).")
    if seconds is not None and seconds > settings.max_audio_seconds:
        raise AudioRejected(413, f"Recording is {seconds:.0f} s long (limit {settings.max_audio_seconds} s).")
    return mime


def decode_16k_mono(data: bytes):
    """Decode any supported container to 16 kHz mono float32 ourselves (PyAV), instead of relying on faster-whisper's
    decoder, whose PyAV call signature breaks across PyAV releases."""
    import av
    import numpy as np

    out = []
    with av.open(io.BytesIO(data)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):  # flush
            out.append(f.to_ndarray().reshape(-1))
    if not out:
        return np.zeros(0, dtype=np.float32)
    return (np.concatenate(out).astype(np.float32) / 32768.0)


@lru_cache(maxsize=1)
def vocabulary() -> List[str]:
    data, lex = drug_data(), lexicon()
    words = set(MEDICAL_TERMS) | set(data["anticoagulants"]) | {d for ds in data["allergy_classes"].values() for d in ds}
    for exps in list(lex["abbreviations"].values()) + list(lex["lay_terms"].values()):
        words.update(exps)
    return sorted(w for w in words if len(w) >= 5 and "/" not in w)


def correct_low_confidence(words: List[dict], threshold: float) -> List[dict]:
    """Replace runs of low-confidence words that closely match a vocabulary term ("a pixaban" -> "apixaban")."""
    vocab = vocabulary()
    flat = {v.replace(" ", ""): v for v in vocab}
    corrections = []
    i = 0
    while i < len(words):
        if words[i]["probability"] >= threshold:
            i += 1
            continue
        best = None
        for span in (3, 2, 1):  # longest span first: split drug names are common
            chunk = words[i:i + span]
            if len(chunk) < span:
                continue
            raw = "".join(re.sub(r"[^a-z]", "", w["word"].lower()) for w in chunk)
            if len(raw) < 5:
                continue
            match = difflib.get_close_matches(raw, flat.keys(), n=1, cutoff=0.82)
            if match:
                best = (span, flat[match[0]], difflib.SequenceMatcher(None, raw, match[0]).ratio())
                break
        if best:
            span, term, ratio = best
            original = " ".join(w["word"].strip() for w in words[i:i + span])
            if original.lower().strip(" .,") != term:
                corrections.append({"from": original, "to": term, "similarity": round(ratio, 3), "index": i})
            i += span
        else:
            i += 1
    return corrections


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: Optional[float]
    words: List[dict]  # {word, start, end, probability, char_start, char_end, low}
    segments: List[dict]
    corrections: List[dict] = field(default_factory=list)
    engine: str = ""

    @property
    def confidence(self) -> Optional[float]:
        return round(sum(w["probability"] for w in self.words) / len(self.words), 3) if self.words else None

    @property
    def low_confidence_words(self) -> List[dict]:
        return [w for w in self.words if w["low"]]


def _finalise(words_raw: List[dict], segments: List[dict], language: str, duration: Optional[float], engine: str, correct: bool = True) -> Transcript:
    threshold = settings.stt_min_word_confidence
    corrections = correct_low_confidence(words_raw, threshold) if correct else []
    by_index = {c["index"]: c for c in corrections}
    out_words, parts, i, pos = [], [], 0, 0
    while i < len(words_raw):
        if i in by_index:
            c = by_index[i]
            span = len(c["from"].split())
            group = words_raw[i:i + span]
            token = (" " if parts else "") + c["to"]
            prob = min(w["probability"] for w in group)
            out_words.append({"word": c["to"], "start": group[0]["start"], "end": group[-1]["end"], "probability": prob,
                              "char_start": pos + (1 if parts else 0), "char_end": pos + len(token), "low": True, "corrected_from": c["from"]})
            parts.append(token)
            pos += len(token)
            i += span
            continue
        w = words_raw[i]
        token = w["word"] if parts else w["word"].lstrip()
        start = pos + (len(token) - len(token.lstrip()))
        out_words.append({**w, "char_start": start, "char_end": pos + len(token), "low": w["probability"] < threshold})
        parts.append(token)
        pos += len(token)
        i += 1
    return Transcript("".join(parts).strip(), language, duration, out_words, segments, corrections, engine)


class FasterWhisperTranscriber:
    def __init__(self, model: str, compute_type: str):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="cpu", compute_type=compute_type, download_root=settings.model_cache_dir)
        self.name = f"faster-whisper:{model}:{compute_type}"

    def transcribe(self, data: bytes, language: Optional[str] = None, prompt: bool = True, correct: bool = True) -> Transcript:
        lang = language or settings.stt_language
        initial = ("Emergency department triage. Medications and conditions: " + ", ".join(MEDICAL_TERMS) + ".") if prompt else None
        segments, info = self.model.transcribe(
            decode_16k_mono(data), language=None if lang == "auto" else lang, initial_prompt=initial, word_timestamps=True,
            vad_filter=True, beam_size=5, temperature=[0.0, 0.2, 0.4], condition_on_previous_text=False,
        )
        words, segs = [], []
        for s in segments:
            segs.append({"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob})
            for w in s.words or []:
                words.append({"word": w.word, "start": w.start, "end": w.end, "probability": float(w.probability)})
        return _finalise(words, segs, info.language, info.duration, self.name, correct)


@lru_cache(maxsize=2)
def get_transcriber(model: Optional[str] = None):
    if settings.stt_engine == "none":
        return None
    try:
        return FasterWhisperTranscriber(model or settings.stt_model, settings.stt_compute_type)
    except ImportError:
        logger.warning("faster-whisper is not installed (pip install -r backend/requirements-multimodal.txt); server dictation disabled.")
        return None


def wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """Word error rate via Levenshtein over lowercased, punctuation-free tokens."""
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()  # noqa: E731
    r, h = norm(reference), norm(hypothesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return {"wer": d[len(h)] / max(len(r), 1), "errors": d[len(h)], "ref_words": len(r)}
