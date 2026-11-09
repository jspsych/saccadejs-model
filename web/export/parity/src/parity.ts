import { OrtEmbeddingModel } from "./model";
import { EMB_DIM, EYE_H, EYE_W } from "./types";

interface Npy {
  data: Float32Array | Uint8Array;
  shape: number[];
}

export interface ParityReport {
  ep: "webgpu" | "wasm";
  n: number;
  perSampleMaxAbs: number[];
  maxAbs: number;
  meanAbs: number;
  msPerEmbed: number;
  graphCapture: boolean;
  medianMs: number;
  p90Ms: number;
}

const BENCH_WARMUP = 10;
const BENCH_ITERS = 50;

async function benchmark(model: OrtEmbeddingModel, crop: Uint8Array): Promise<[number, number]> {
  for (let i = 0; i < BENCH_WARMUP; i++) await model.embed(crop);
  const samples: number[] = [];
  for (let i = 0; i < BENCH_ITERS; i++) {
    const t = performance.now();
    await model.embed(crop);
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  return [samples[Math.floor(samples.length / 2)], samples[Math.floor(samples.length * 0.9)]];
}

function parseNpy(buf: ArrayBuffer): Npy {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(...bytes.subarray(1, 6));
  if (bytes[0] !== 0x93 || magic !== "NUMPY") throw new Error("not a .npy file");
  const major = bytes[6];
  const view = new DataView(buf);
  let headerLen: number;
  let offset: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    offset = 10 + headerLen;
  } else {
    headerLen = view.getUint32(8, true);
    offset = 12 + headerLen;
  }
  const header = new TextDecoder().decode(bytes.subarray(offset - headerLen, offset));
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeStr = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  if (!descr) throw new Error("npy header missing descr");
  if (fortran === "True") throw new Error("fortran_order npy not supported");
  const shape = shapeStr
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  if (descr === "<f4") return { data: new Float32Array(buf.slice(offset, offset + count * 4)), shape };
  if (descr === "|u1" || descr === "u1" || descr === "<u1")
    return { data: new Uint8Array(buf.slice(offset, offset + count)), shape };
  throw new Error(`unsupported dtype ${descr}`);
}

async function fetchNpy(url: string): Promise<Npy> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if ((res.headers.get("content-type") ?? "").startsWith("text/html"))
    throw new Error(`${url}: not found (dev server returned index.html)`);
  return parseNpy(await res.arrayBuffer());
}

function log(line: string): void {
  const out = document.getElementById("out");
  if (out) out.textContent += line + "\n";
  console.log(line);
}

async function runOne(
  ep: "webgpu" | "wasm",
  crops: Uint8Array[],
  expected: Float32Array,
): Promise<ParityReport> {
  const model = new OrtEmbeddingModel({ executionProviders: [ep] });
  const actual = await model.init();
  if (actual.ep !== ep) throw new Error(`requested ${ep}, got ${actual.ep}`);
  await model.embed(crops[0]);
  const perSampleMaxAbs: number[] = [];
  let maxAbs = 0;
  let sumAbs = 0;
  const t0 = performance.now();
  for (let i = 0; i < crops.length; i++) {
    const e = await model.embed(crops[i]);
    let m = 0;
    for (let j = 0; j < EMB_DIM; j++) {
      const d = Math.abs(e[j] - expected[i * EMB_DIM + j]);
      if (d > m) m = d;
      sumAbs += d;
    }
    perSampleMaxAbs.push(m);
    if (m > maxAbs) maxAbs = m;
  }
  const msPerEmbed = (performance.now() - t0) / crops.length;
  const [medianMs, p90Ms] = await benchmark(model, crops[0]);
  model.dispose();
  return {
    ep,
    n: crops.length,
    perSampleMaxAbs,
    maxAbs,
    meanAbs: sumAbs / (crops.length * EMB_DIM),
    msPerEmbed,
    graphCapture: model.graphCapture,
    medianMs,
    p90Ms,
  };
}

export async function runParity(): Promise<ParityReport[]> {
  const out = document.getElementById("out");
  if (out) out.textContent = "";
  const inputs = await fetchNpy("/fixtures/model_inputs.npy");
  const embeddings = await fetchNpy("/fixtures/model_embeddings.npy");
  const n = inputs.shape[0];
  const len = EYE_H * EYE_W;
  if (inputs.data.length !== n * len) throw new Error(`unexpected input shape ${inputs.shape}`);
  if (embeddings.data.length !== n * EMB_DIM)
    throw new Error(`unexpected embedding shape ${embeddings.shape}`);
  const expected = Float32Array.from(embeddings.data);
  const crops: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    const c = new Uint8Array(len);
    for (let j = 0; j < len; j++) c[j] = Math.round(inputs.data[i * len + j]);
    crops.push(c);
  }
  log(`fixtures: ${n} samples, input shape [${inputs.shape}], embedding shape [${embeddings.shape}]`);

  const reports: ParityReport[] = [];
  for (const ep of ["webgpu", "wasm"] as const) {
    log("");
    try {
      const r = await runOne(ep, crops, expected);
      reports.push(r);
      log(`ep=${r.ep}  n=${r.n}  maxAbs=${r.maxAbs.toExponential(3)}  meanAbs=${r.meanAbs.toExponential(3)}  ${r.msPerEmbed.toFixed(2)} ms/embed`);
      r.perSampleMaxAbs.forEach((v, i) => log(`  sample ${String(i).padStart(3)}  ${v.toExponential(3)}`));
      log(
        `  bench ep=${r.ep} graphCapture=${r.graphCapture} warmup=${BENCH_WARMUP} n=${BENCH_ITERS}  median=${r.medianMs.toFixed(2)} ms  p90=${r.p90Ms.toFixed(2)} ms`,
      );
    } catch (err) {
      log(`ep=${ep} failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return reports;
}

runParity().catch((err) => {
  console.error(err);
  log(`error: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});
