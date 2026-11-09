// Provenance: a frozen copy of `packages/core/src/types.ts` in jspsych/saccadejs, trimmed
// to what the parity page needs (EYE_W/EYE_H/EMB_DIM and the model interface).
//
export const EYE_H = 36;
export const EYE_W = 144;
export const EMB_DIM = 128;

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface BBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface EyeCrop {
  data: Uint8Array;
  bbox: BBox;
}

export type Embedding = Float32Array;

export interface Gaze {
  x: number;
  y: number;
}

export interface EmbeddingModel {
  init(): Promise<{ ep: "webgpu" | "wasm" }>;
  embed(crop: Uint8Array): Promise<Float32Array>;
}

export interface CalPoint {
  target: Gaze;
  embeddings: Float32Array[];
  meanEmbedding: Float32Array;
}

export interface CalHead {
  kernel: number[];
  bias: number;
}

export interface RidgeRow {
  e: Float32Array;
  x: number;
  y: number;
  w: number;
}
