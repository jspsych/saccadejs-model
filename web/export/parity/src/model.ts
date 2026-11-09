// Provenance: a frozen copy of the ONNX wrapper from `packages/core/src/model.ts` in
// jspsych/saccadejs. It is deliberately minimal: what this page validates is the exported
// .onnx running under onnxruntime-web on WebGPU, not the wrapper. check_parity_ort.py
// covers the CPU provider; this page is the only check of the GPU path that ships.
//
import * as ort from "onnxruntime-web";
import ortWasmMjs from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url";
import ortWasmBin from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url";
import { EMB_DIM, EYE_H, EYE_W } from "./types";
import type { EmbeddingModel } from "./types";

export interface OrtModelOptions {
  modelPath?: string;
  executionProviders?: ("webgpu" | "wasm")[];
}

const CROP_LEN = EYE_H * EYE_W;

// Pinned to export_onnx.py's IN_NAME/OUT_NAME, which rename_io() forces onto the graph.
// Deliberately a constant rather than a read of the generated export manifest: that kept
// this page from typechecking in a fresh clone (out/ is gitignored), and a mismatch here
// should be an error rather than a silent rebind to inputNames[0] -- on a parity page,
// quietly accepting whatever the model exposes hides the discrepancy it exists to find.
const INPUT_NAME = "eye_image";
const OUTPUT_NAME = "embedding";

export class OrtEmbeddingModel implements EmbeddingModel {
  readonly modelPath: string;
  private eps: ("webgpu" | "wasm")[];
  private capture = false;
  private session: ort.InferenceSession | null = null;
  private input!: ort.Tensor;
  private inputData!: Float32Array;
  private inputName = INPUT_NAME;
  private outputName = OUTPUT_NAME;
  private ep: "webgpu" | "wasm" = "wasm";

  constructor(opts: OrtModelOptions = {}) {
    this.modelPath = opts.modelPath ?? "/models/eye_embedding.onnx";
    this.eps = opts.executionProviders ?? ["webgpu", "wasm"];
  }

  /** True when the WebGPU session was created with (and survived) graph capture. */
  get graphCapture(): boolean {
    return this.capture;
  }

  async init(): Promise<{ ep: "webgpu" | "wasm" }> {
    // Vite refuses dynamic import() of files under public/, so hand ORT
    // bundler-resolved URLs instead of a wasmPaths directory.
    ort.env.wasm.wasmPaths = { mjs: ortWasmMjs, wasm: ortWasmBin };
    ort.env.wasm.numThreads = 1;

    this.inputData = new Float32Array(CROP_LEN);
    this.input = new ort.Tensor("float32", this.inputData, [1, EYE_H, EYE_W, 1]);

    let lastErr: unknown = null;
    for (const ep of this.eps) {
      // Graph capture in ORT-web requires every input/output to be a
      // pre-allocated 'gpu-buffer' tensor; with a CPU input it throws at the
      // first run(), so the warm-up is what decides whether it stays on.
      for (const capture of ep === "webgpu" ? [true, false] : [false]) {
        try {
          this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: [ep],
            graphOptimizationLevel: "all",
            ...(capture ? { enableGraphCapture: true } : {}),
          });
          this.assertNames();
          await this.embed(new Uint8Array(CROP_LEN));
          this.ep = ep;
          this.capture = capture;
          lastErr = null;
          break;
        } catch (err) {
          lastErr = err;
          this.session?.release();
          this.session = null;
        }
      }
      if (this.session) break;
    }
    if (!this.session) throw lastErr ?? new Error("no execution provider available");
    return { ep: this.ep };
  }

  private assertNames(): void {
    if (!this.session) return;
    const ins = this.session.inputNames;
    const outs = this.session.outputNames;
    if (!ins.includes(this.inputName)) {
      throw new Error(
        `model input ${this.inputName} not found; graph exposes [${ins.join(", ")}]. ` +
          "Re-run export_onnx.py -- rename_io() should have pinned it.",
      );
    }
    if (!outs.includes(this.outputName)) {
      throw new Error(
        `model output ${this.outputName} not found; graph exposes [${outs.join(", ")}]. ` +
          "Re-run export_onnx.py -- rename_io() should have pinned it.",
      );
    }
  }

  async embed(crop: Uint8Array): Promise<Float32Array> {
    if (!this.session) throw new Error("model not initialized");
    if (crop.length !== CROP_LEN) throw new Error(`crop length ${crop.length} != ${CROP_LEN}`);
    for (let i = 0; i < CROP_LEN; i++) this.inputData[i] = crop[i];
    const out = await this.session.run({ [this.inputName]: this.input });
    const t = out[this.outputName];
    return Float32Array.from(t.data as Float32Array);
  }

  dispose(): void {
    this.session?.release();
    this.session = null;
  }
}

export class StubEmbeddingModel implements EmbeddingModel {
  private static readonly BLOCKS_X = 16;
  private static readonly BLOCKS_Y = 4;
  private weights: Float32Array;

  constructor() {
    const nb = StubEmbeddingModel.BLOCKS_X * StubEmbeddingModel.BLOCKS_Y;
    this.weights = new Float32Array(EMB_DIM * nb);
    let s = 0x9e3779b9;
    for (let i = 0; i < this.weights.length; i++) {
      s = (Math.imul(s ^ (s >>> 15), 0x85ebca6b) + 0x165667b1) >>> 0;
      this.weights[i] = (s / 0xffffffff) * 2 - 1;
    }
  }

  init(): Promise<{ ep: "webgpu" | "wasm" }> {
    return Promise.resolve({ ep: "wasm" });
  }

  embed(crop: Uint8Array): Promise<Float32Array> {
    const bx = StubEmbeddingModel.BLOCKS_X;
    const by = StubEmbeddingModel.BLOCKS_Y;
    const nb = bx * by;
    const means = new Float32Array(nb);
    const bw = EYE_W / bx;
    const bh = EYE_H / by;
    for (let y = 0; y < EYE_H; y++) {
      const yb = Math.min(by - 1, Math.floor(y / bh));
      for (let x = 0; x < EYE_W; x++) {
        const xb = Math.min(bx - 1, Math.floor(x / bw));
        means[yb * bx + xb] += crop[y * EYE_W + x];
      }
    }
    const perBlock = (EYE_W * EYE_H) / nb;
    for (let i = 0; i < nb; i++) means[i] = means[i] / perBlock / 255 - 0.5;

    const out = new Float32Array(EMB_DIM);
    for (let j = 0; j < EMB_DIM; j++) {
      let acc = 0;
      for (let i = 0; i < nb; i++) acc += this.weights[j * nb + i] * means[i];
      out[j] = Math.tanh(acc);
    }
    return Promise.resolve(out);
  }
}
