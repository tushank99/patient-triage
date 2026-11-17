# Multimodal intake (Phase 6)

```
dictation ─► audio validation ─► faster-whisper (medical prompt, word confidence, VAD) ─► vocabulary correction
          ─► symptom / duration / severity / onset / medication / allergy extraction ─► CAPTURE (draft)
scan/photo ─► image validation ─► OCR (preprocess, orientation, PP-OCR) ─► layout + tables ─► normalised fields ─► CAPTURE
CAPTURE ─► AUTO_ACCEPTED (everything confident)  or  PENDING_REVIEW ─► clinician verifies / corrects / rejects
```
Code: `backend/app/intake/`. API: `/api/intake/transcribe`, `/api/captures/*`, scanned files via the normal document upload.

## The verification gate (the important part)
* **Dictation**: any word under `STT_MIN_WORD_CONFIDENCE`, any vocabulary auto-correction, or any low-confidence field
  makes the transcript `PENDING_REVIEW`. A triage request that references it gets **409** until a clinician confirms
  (and optionally edits) it. Edits are stored with an edit distance (correction metric).
* **Scans**: any OCR line under `OCR_MIN_LINE_CONFIDENCE`, any field under `OCR_MIN_FIELD_CONFIDENCE`, an empty page, or a
  printed MRN that differs from the patient it is filed under puts the document in `NEEDS_REVIEW`: **not indexed, not
  searchable, not used for facts or safety rules**. On verify, the clinician's corrected text (not the OCR text) is what
  gets indexed. Reject marks the document `FAILED`.
* High-confidence output is `AUTO_ACCEPTED` and can still be reviewed and corrected later.
* Original audio is not stored (voice is identifying); transcripts and word confidences are.

## Speech-to-text
faster-whisper (CTranslate2, int8 CPU), lazy-loaded. Medical bias via `initial_prompt` (drug and condition names);
low-confidence word runs are fuzzy-matched to the vocabulary ("a pixa ban" → "apixaban"), every substitution shown.
Silero VAD drops silence/noise; temperature fallback retries hard segments. Audio is decoded by our own PyAV code (16 kHz
mono), because faster-whisper 1.2.1's decoder breaks with PyAV 19. Accepted: WAV, WebM (browser MediaRecorder), Ogg,
MP3, M4A, FLAC; ≤ `MAX_AUDIO_MB`, ≤ `MAX_AUDIO_SECONDS`. Retries with the same `request_id` return the same capture.
`STT_LANGUAGE` sets the language (`auto` detects). If server STT is off, the UI falls back to browser dictation.

## OCR
PaddleOCR's PP-OCR models, run through RapidOCR on ONNX Runtime by default (~15 MB of models, no paddlepaddle);
`OCR_ENGINE=paddleocr` uses native PaddleOCR when installed. Preprocessing: upscale small scans, denoise, CLAHE contrast,
deskew (Hough). Orientation: the recogniser reads sideways and upside-down text with high confidence, so orientation
is chosen by layout (horizontal text boxes, title in the top half), not by confidence. Layout: reading order with
column detection; tables rebuilt from aligned rows. Fields: labs (canonical name, value, unit, normalised value,
reference range, H/L), medications, allergies, dates (ISO; ambiguous day/month lowers confidence, `DATE_ORDER`),
MRN and printed name. Units: °F→°C, lb→kg, glucose mg/dL→mmol/L, creatinine mg/dL→µmol/L, Hb g/L→g/dL.
Images: PNG, JPEG, TIFF, size-limited, decompression-bomb guarded (`MAX_IMAGE_PIXELS`); scanned PDFs are rendered at 200 dpi.

## Benchmarks
* `python -m backend.eval.ocr_benchmark --report docs/OCR_RESULTS.md`: synthetic lab reports and medication lists,
  six degradations; CER, field accuracy, and the gate's safety numbers (wrong-but-auto-accepted, errors held for review).
* `python -m backend.eval.stt_benchmark --report docs/STT_RESULTS.md`: clinical complaints spoken by the Windows TTS voices,
  clean and noisy; WER, medical-term recall, extraction accuracy, gate catch rate, real-time factor.
Both use synthetic inputs (rendered text, TTS speech) and overstate real-world accuracy. Collect real, consented
recordings and scans before trusting absolute numbers.

## Deployment and memory
Engines are optional (`backend/requirements-multimodal.txt`). Whisper base.en int8 adds ~150–250 MB and OCR ~150 MB on
top of the backend and embedding model, which exceeds Render's 512 MB free tier. Options: set `STT_ENGINE=none` and
`OCR_ENGINE=none` there (browser dictation; scanned uploads rejected with a clear message), or run them on a larger
instance / separate worker (Phase 7).
