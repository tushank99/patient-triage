/* eslint-disable prettier/prettier */
import { EsiLevel } from "./triage-context";

export interface EdgeModelInput {
  age: number | null;
  hr: number | null;
  bpSys: number | null;
  bpDia: number | null;
  spo2: number | null;
  temp: number | null;
}

export async function runEdgeInference(vitals: EdgeModelInput) {
  // Simulate WebAssembly (Wasm) loading and tensor inference time
  await new Promise(resolve => setTimeout(resolve, 600));

  let esi: EsiLevel = 4;
  
  
  let baseConfidence = 85; 
  
  const badges: {label: string, type: "danger"|"warning"|"success" | "default"}[] = [
    { label: " Edge ML Active", type: "warning" },
    { label: " Cloud Disconnected", type: "danger" }
  ];
  
  const flags: string[] = [];

  // Deterministic local inference rules (Simulating the ONNX output)
  if (vitals.age !== null && vitals.age <= 0.25 && vitals.temp !== null && vitals.temp >= 38.0) {
    esi = 2; 
    flags.push("Neonatal fever protocol");
  } else if (vitals.spo2 !== null && vitals.spo2 <= 92) {
    esi = 2; 
    flags.push(`Hypoxia (SpO2 ${vitals.spo2}%)`);
  } else if (vitals.hr !== null && vitals.hr > 120) {
    esi = 3; 
    flags.push(`Tachycardia (HR ${vitals.hr})`);
  } else if (vitals.bpSys !== null && vitals.bpSys > 160) {
    esi = 3; 
    flags.push(`Hypertension (Sys ${vitals.bpSys})`);
  }

  // Edge Case 1: Imputation Penalty
  // Mathematically penalize confidence if tensors are empty
  if (vitals.hr === null) baseConfidence -= 7;
  if (vitals.spo2 === null) baseConfidence -= 7;
  if (vitals.bpSys === null) baseConfidence -= 7;

  // Edge Case 4: Rationale Degradation
  // We cannot generate fluent LLM text, so we output rigid telemetry flags.
  const rationale = flags.length > 0
    ? `Cloud NLP Rationale Unavailable. Local Engine Flags: ${flags.join(", ")}.`
    : `Cloud NLP Rationale Unavailable. Local Engine: Vitals stable within normal physiological bounds.`;

  return {
    recommended_esi: esi,
    confidence_score: Math.max(60, baseConfidence), 
    rationale,
    badges
  };
}