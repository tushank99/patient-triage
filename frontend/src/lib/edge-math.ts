/**
 * The model maths that must match the server exactly, kept FREE of imports so Node can run it in the
 * parity check (scripts/verify-onnx-parity.mjs) against vectors exported by the backend:
 *   - feature vector construction, incl. derived "<vital>_missing" indicators, in the meta's order
 *   - temperature calibration  softmax(log(p) / T)
 *   - decoding class index -> ESI via the meta's class_to_esi (never assume index+1)
 * Python counterparts: backend/app/ml/feature_schema.py (build_feature_vector) and ml/model.py (apply_temperature).
 */

export interface FeatureSpec {
  name: string;
  derived_from?: string;
  kind?: string;
}

export interface ModelMeta {
  model_version: string;
  label_source?: string;
  feature_schema_version: string;
  features: FeatureSpec[];
  class_to_esi: number[];
  calibration?: { method: string; temperature: number };
}

export type RawInput = Record<string, number | boolean | null | undefined>;

const isNull = (v: unknown): boolean => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));

export function buildFeatureVector(meta: ModelMeta, input: RawInput): Float32Array {
  return Float32Array.from(
    meta.features.map((f) => {
      if (f.kind === "is_missing" && f.derived_from) return isNull(input[f.derived_from]) ? 1 : 0;
      const v = input[f.name];
      if (typeof v === "boolean") return v ? 1 : 0;
      return isNull(v) ? Number.NaN : (v as number);
    }),
  );
}

/** softmax(log(p) / T); identity when T === 1. Never changes the argmax. */
export function applyTemperature(probs: ArrayLike<number>, temperature: number): number[] {
  const p = Array.from(probs);
  if (temperature === 1) return p;
  const q = p.map((x) => Math.pow(Math.min(Math.max(x, 1e-12), 1), 1 / temperature));
  const sum = q.reduce((a, b) => a + b, 0);
  return q.map((x) => x / sum);
}

export function argmax(arr: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if ((arr[i] as number) > (arr[best] as number)) best = i;
  return best;
}

export function decode(meta: ModelMeta, rawProbs: ArrayLike<number>): { probs: number[]; esi: number; top: number } {
  const probs = applyTemperature(Array.from(rawProbs).slice(0, meta.class_to_esi.length), meta.calibration?.temperature ?? 1);
  const k = argmax(probs);
  return { probs, esi: meta.class_to_esi[k] as number, top: probs[k] as number };
}
