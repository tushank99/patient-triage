/* eslint-disable prettier/prettier */
/**
 * Edge AI Fallback Engine — runs the same XGBoost model locally via ONNX + WebAssembly.
 *
 * FEATURE ORDER SPEC (must match backend/app/services/assessment_engine.py):
 *   [age, hr, bpSys, bpDia, spo2, temp, pain, has_mrn]
 *
 * Defaults for missing values use the same clinical defaults as the backend:
 *   age=35, hr=75, bpSys=120, bpDia=80, spo2=98, temp=37.0, pain=0, has_mrn=0
 */
import { EsiLevel } from "./triage-context";
import * as ort from "onnxruntime-web";

// Configure ONNX Runtime WebAssembly path
ort.env.wasm.wasmPaths = "/";

export interface EdgeModelInput {
  age: number | null;
  hr: number | null;
  bpSys: number | null;
  bpDia: number | null;
  spo2: number | null;
  temp: number | null;
  pain?: number | null;
  hasMrn?: boolean;
}

// Cache the ONNX session so the .onnx file is only fetched once
let cachedSession: ort.InferenceSession | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!cachedSession) {
    cachedSession = await ort.InferenceSession.create("/triage_model.onnx");
  }
  return cachedSession;
}

/**
 * Build the Float32 feature vector in the exact order the Python model was trained on.
 * Missing values are imputed with the same clinical defaults used on the backend.
 */
function toFeatureVector(vitals: EdgeModelInput): Float32Array {
  return Float32Array.from([
    vitals.age ?? 35.0,
    vitals.hr ?? 75.0,
    vitals.bpSys ?? 120.0,
    vitals.bpDia ?? 80.0,
    vitals.spo2 ?? 98.0,
    vitals.temp ?? 37.0,
    vitals.pain ?? 0.0,
    vitals.hasMrn ? 1.0 : 0.0,
  ]);
}

/**
 * Return the index of the largest element (argmax).
 */
function argmax(arr: ArrayLike<number>): number {
  let maxIdx = 0;
  let maxVal = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
  }
  return maxIdx;
}

export async function runEdgeInference(vitals: EdgeModelInput) {
  const badges: { label: string; type: "danger" | "warning" | "success" | "default" }[] = [
    { label: "⚡ Edge ONNX Active", type: "warning" },
    { label: "☁️ Cloud Disconnected", type: "danger" },
  ];

  // --- Confidence penalty for missing vitals (same logic as backend) ---
  let missingPenalty = 0;
  if (vitals.hr === null) missingPenalty += 7;
  if (vitals.spo2 === null) missingPenalty += 7;
  if (vitals.bpSys === null) missingPenalty += 7;

  // --- Run real ONNX inference ---
  const session = await getSession();

  const inputTensor = new ort.Tensor("float32", toFeatureVector(vitals), [1, 8]);

  // Feed the tensor using the model's actual input name
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);

  // The sklearn ONNX converter outputs: "output_label" (predicted class) and
  // "output_probability" (per-class probabilities as a sequence of maps).
  // We try to use probabilities first for confidence, fall back to label-only.
  let esi: EsiLevel = 4;
  let baseConfidence = 85;

  const outputNames = session.outputNames;

  // Try to extract the predicted label
  const labelOutput = results[outputNames[0]];
  const predictedClass = Number(labelOutput.data[0]);
  if (predictedClass >= 1 && predictedClass <= 5) {
    esi = predictedClass as EsiLevel;
  }

  // Try to extract probability from the second output (if available)
  if (outputNames.length > 1) {
    const probOutput = results[outputNames[1]];
    if (probOutput && probOutput.data) {
      // For XGBoost ONNX models the probabilities output is a sequence of maps.
      // onnxruntime-web may flatten this — we try to extract the max probability.
      const probData = probOutput.data as Float32Array;
      if (probData.length >= 5) {
        const maxProb = Math.max(...Array.from(probData).slice(0, 5));
        baseConfidence = Math.round(maxProb * 100);
      }
    }
  }

  // Apply missing-data penalty
  const confidence = Math.max(50, baseConfidence - missingPenalty);

  const flags: string[] = [`ONNX Tensor Prediction: ESI ${esi}`];

  // Rationale — we cannot produce fluent LLM text offline, so output rigid telemetry
  const rationale =
    `Cloud NLP Rationale Unavailable. Local ONNX Engine Flags: ${flags.join(", ")}.`;

  return {
    recommended_esi: esi,
    confidence_score: confidence,
    rationale,
    badges,
  };
}