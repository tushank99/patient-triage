/**
 * Proves the BROWSER inference path equals the SERVER's.
 *
 * Loads the exact bundle the app ships (public/models/triage_model.onnx + meta), builds feature vectors with the
 * app's own edge-math.ts, runs onnxruntime-web (the engine the browser uses), applies the app's calibration and
 * decoding, and compares against the probabilities the backend computed from its joblib model for the same inputs
 * (public/models/parity_vectors.json, written by backend/training).
 *
 *   node --experimental-strip-types scripts/verify-onnx-parity.mjs     (Node 22.6+; default in 23.6+)
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { buildFeatureVector, decode } from "../src/lib/edge-math.ts";

const dir = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "models");
const meta = JSON.parse(readFileSync(join(dir, "triage_model.meta.json"), "utf8"));
const { vectors, model_version: vectorsFor } = JSON.parse(readFileSync(join(dir, "parity_vectors.json"), "utf8"));

if (vectorsFor !== meta.model_version) {
  console.error(`parity_vectors.json is for ${vectorsFor} but the bundle is ${meta.model_version}. Re-run promote.`);
  process.exit(1);
}

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(readFileSync(join(dir, "triage_model.onnx")));
const inputName = session.inputNames[0];

let maxDiff = 0;
let esiMismatches = 0;
for (const v of vectors) {
  const x = buildFeatureVector(meta, v.input);
  const out = await session.run({ [inputName]: new ort.Tensor("float32", x, [1, x.length]) });
  const d = decode(meta, out["probabilities"].data);
  v.probabilities.forEach((p, i) => { maxDiff = Math.max(maxDiff, Math.abs(p - d.probs[i])); });
  if (d.esi !== v.esi) esiMismatches++;
}

const TOL = 1e-4;
console.log(`browser-vs-server parity: ${vectors.length} vectors, max |dp| = ${maxDiff.toExponential(2)}, ESI mismatches = ${esiMismatches}`);
if (maxDiff > TOL || esiMismatches > 0) {
  console.error(`FAIL: tolerance is ${TOL} and 0 mismatches.`);
  process.exit(1);
}
console.log("OK");
