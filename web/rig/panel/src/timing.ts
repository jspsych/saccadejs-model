// Screen -> webcam timing loopback page.
//
// Flashes the whole screen in an m-sequence of black/white symbols, records the webcam's mean
// luminance per frame, and cross-correlates the two. The peak lag is display lag + camera lag:
// the constant to subtract from a gaze sample's `captureTime` to align it with stimulus onset.
//
// All the maths lives in ./loopback (DOM-free, unit-tested). This file is only plumbing.

import {
  estimateLag,
  estimateLagEdges,
  intervalStats,
  mSequence,
  seededRandom,
  sparseSchedule,
  splitHalves,
} from "./loopback";
import type { Flip, IntervalStats, LagResult, LumSample } from "./loopback";
import { Rig } from "./rig";
import type { RigEdgeEvent, RigEvent, RigInfo, RigLedEvent } from "./rig";
import {
  chooseLedSignal,
  clusterLags,
  displayLag,
  exposureWindow,
  fitSync,
  relayResiduals,
  roiMean,
  rowBands,
} from "./rigAnalysis";
import type {
  CamFrame,
  CamFrameGray,
  DisplayLagResult,
  ExposureResult,
  LagStats,
  LedPlane,
  LedSource,
  RelayResult,
  RigEdge,
  RigLed,
  RoiResult,
  RowBandResult,
  SyncFit,
  SyncPair,
} from "./rigAnalysis";

const SAMPLE_W = 64;
const SAMPLE_H = 48;
/** The LED detector's frame: big enough that a 5 mm LED is more than one pixel. */
const PEAK_W = 160;
const PEAK_H = 120;
/** How many of the brightest pixels the LED detector averages. */
const PEAK_N = 24;
const COUNTDOWN_S = 3;
const CURVE_W = 600;
const CURVE_H = 160;
const RIG_W = 600;
const RIG_H = 160;
const SYNC_PINGS = 20;
/**
 * Verdict thresholds for the closed-loop test: how far the LED plateau may sit from the
 * loopback estimate, and how many pulses must agree on one lag for the plateau to mean
 * anything.
 */
const MAX_RIG_RESIDUAL_MS = 5;
const MIN_RIG_CONSISTENT_FRACTION = 0.7;
/** Rows of the delay -> residual table in the report. */
const MAX_RESIDUAL_ROWS = 20;

const MIN_PEAK_CORR = 0.6;
const MIN_PEAK_D = 0.5;
const MAX_SECOND_PEAK_RATIO = 0.7;
// Each half has only ~5 s of data; the estimator's own tail on that much data is ~10 ms.
const MAX_HALF_DISAGREEMENT_MS = 20;
/** With this many edges per half the estimate is well determined, so hold it to 8 ms. */
const DENSE_HALF_MIN_EDGES = 15;
const MAX_HALF_DISAGREEMENT_DENSE_MS = 8;
/** One camera frame at 30 fps: the plateau must be narrower than this to have sub-frame resolution. */
const MAX_PLATEAU_WIDTH_MS = 34;

// ---------------------------------------------------------------- rVFC typings
// requestVideoFrameCallback is not in lib.dom, so declare the minimal shape we use.

interface VideoFrameMeta {
  captureTime?: number;
  receiveTime?: number;
  expectedDisplayTime?: number;
  presentedFrames?: number;
}

type RvfcCallback = (now: number, metadata: VideoFrameMeta) => void;

interface RvfcVideoElement {
  requestVideoFrameCallback(cb: RvfcCallback): number;
  cancelVideoFrameCallback(handle: number): void;
}

function asRvfc(v: HTMLVideoElement): RvfcVideoElement | null {
  const c = v as unknown as Partial<RvfcVideoElement>;
  return typeof c.requestVideoFrameCallback === "function" ? (c as RvfcVideoElement) : null;
}

// ---------------------------------------------------------------- record types

interface FlipRecord {
  /** rAF timestamp of the callback that painted this level. */
  t: number;
  /** performance.now() inside that callback, for reference. */
  tNow: number;
  level: number;
}

interface CamSample {
  now: number;
  captureTime: number | null;
  receiveTime: number | null;
  expectedDisplayTime: number | null;
  presentedFrames: number | null;
  lum: number;
  /** Mean of the PEAK_N brightest gray pixels: the rig's LED, which the whole-frame mean loses. */
  peak: number;
  /** The same on the redness plane (R - max(G, B)), where a white highlight cannot compete. */
  peakRed: number;
}

type Mode = "sparse" | "mseq";

interface Settings {
  mode: Mode;
  contrast: "full" | "reduced";
  levels: [string, string];
  durationS: number;
  /** sparse mode */
  gapMinMs: number | null;
  gapMaxMs: number | null;
  seed: number | null;
  /** m-sequence mode */
  symbolLen: number | null;
  order: number | null;
  sequenceLength: number | null;
  symbolMsNominal: number | null;
  rvfc: boolean;
  videoWidth: number;
  videoHeight: number;
  exposureMode: string | null;
  frameRate: number | null;
}

interface VariantSummary {
  clock: string;
  lagMs: number;
  lagRefinedMs: number;
  peakCorr: number;
  secondPeakRatio: number;
  plateauCenterMs: number;
  plateauWidthMs: number;
  nSamples: number;
}

interface Results {
  variants: VariantSummary[];
  halves: { first: VariantSummary; second: VariantSummary } | null;
  cameraInterval: IntervalStats;
  rafInterval: IntervalStats;
  droppedCameraFrames: number;
  nFlips: number;
  verdict: string;
}

// ---------------------------------------------------------------- dom helpers

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} missing`);
  return e as T;
}

const startBtn = el<HTMLButtonElement>("start");
const copyBtn = el<HTMLButtonElement>("copy");
const statusEl = el<HTMLSpanElement>("status");
const outEl = el<HTMLPreElement>("out");
const jsonEl = el<HTMLTextAreaElement>("json");
const curveEl = el<HTMLCanvasElement>("curve");
const video = el<HTMLVideoElement>("cam");

function setStatus(s: string): void {
  statusEl.textContent = s;
}

// ---------------------------------------------------------------- basic / advanced
//
// Basic mode is three buttons and a summary card: enough to run every diagnostic without
// touching a setting. Advanced reveals the controls the presets write, the full monospace
// report, the plots and the JSON — nothing is basic-only, so "advanced" only ever adds.

const ADVANCED_KEY = "timing.advanced";
const advancedToggle = el<HTMLInputElement>("advanced");
let advanced = false;

function setAdvanced(on: boolean): void {
  advanced = on;
  advancedToggle.checked = on;
  for (const node of document.querySelectorAll<HTMLElement>(".advanced-only")) node.hidden = !on;
  try {
    localStorage.setItem(ADVANCED_KEY, on ? "1" : "0");
  } catch {
    /* private mode / storage disabled: the choice simply does not persist */
  }
  syncMode();
}

function initAdvanced(): void {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(ADVANCED_KEY);
  } catch {
    stored = null;
  }
  setAdvanced(stored === "1");
  advancedToggle.addEventListener("change", () => {
    setAdvanced(advancedToggle.checked);
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

// ---------------------------------------------------------------- formatting

function num(v: number, d = 2): string {
  return Number.isFinite(v) ? v.toFixed(d) : "n/a";
}

function pad(s: string, w: number): string {
  return s.length >= w ? s : s + " ".repeat(w - s.length);
}

function padL(s: string, w: number): string {
  return s.length >= w ? s : " ".repeat(w - s.length) + s;
}

function summarise(clock: string, r: LagResult): VariantSummary {
  return {
    clock,
    lagMs: r.lagMs,
    lagRefinedMs: r.lagRefinedMs,
    peakCorr: r.peakCorr,
    secondPeakRatio: r.secondPeakRatio,
    plateauCenterMs: r.plateau.centerMs,
    plateauWidthMs: r.plateau.widthMs,
    nSamples: r.nSamples,
  };
}

/** The headline estimate for a variant: plateau center in sparse mode, refined argmax otherwise. */
function estimateOf(v: VariantSummary, mode: Mode): number {
  return mode === "sparse" ? v.plateauCenterMs : v.lagRefinedMs;
}

function statsLine(s: IntervalStats): string {
  return `mean ${num(s.mean)} ms  sd ${num(s.sd)}  min ${num(s.min)}  max ${num(s.max)}  n ${s.n}`;
}

// ---------------------------------------------------------------- LED planes
//
// One 160x120 sample of the video, reduced to two planes: gray, and "redness"
// (R - max(G, B)). The rig's LED is red, so redness rejects a white desk lamp or a specular
// highlight on a face outright, while gray wins for an LED bright enough to bloom to white
// at its centre. Both are computed in one pass and the analysis picks the better one.
// Shared by the capture sampler and the aiming preview so they cannot drift apart.

const PEAK_PIX = PEAK_W * PEAK_H;

interface PlaneSampler {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  gray: Uint8Array;
  red: Uint8Array;
  hist: Int32Array;
  image: ImageData;
}

let planeSampler: PlaneSampler | null = null;

function getPlaneSampler(): PlaneSampler {
  if (planeSampler) return planeSampler;
  const canvas = document.createElement("canvas");
  canvas.width = PEAK_W;
  canvas.height = PEAK_H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d canvas context unavailable");
  planeSampler = {
    canvas,
    ctx,
    gray: new Uint8Array(PEAK_PIX),
    red: new Uint8Array(PEAK_PIX),
    hist: new Int32Array(256),
    image: ctx.createImageData(PEAK_W, PEAK_H),
  };
  return planeSampler;
}

/**
 * Mean of the PEAK_N brightest pixels of a plane. The LED is a few pixels of a 160x120
 * frame — invisible in the whole-frame mean, but it owns the top of the histogram. A
 * 256-bin histogram walked down from the top keeps this O(pixels).
 */
function brightestMean(plane: Uint8Array, hist: Int32Array): number {
  hist.fill(0);
  for (let i = 0; i < plane.length; i++) hist[plane[i]]++;
  let need = PEAK_N;
  let total = 0;
  let got = 0;
  for (let b = 255; b >= 0 && need > 0; b--) {
    const c = hist[b];
    if (c === 0) continue;
    const take = c <= need ? c : need;
    total += b * take;
    got += take;
    need -= take;
  }
  return got > 0 ? total / got : 0;
}

/** Draws the video into the sampler and fills both planes; returns their brightest-N means. */
function samplePlanes(s: PlaneSampler): { peak: number; peakRed: number } {
  s.ctx.drawImage(video, 0, 0, PEAK_W, PEAK_H);
  const d = s.ctx.getImageData(0, 0, PEAK_W, PEAK_H).data;
  for (let i = 0, p = 0; p < PEAK_PIX; i += 4, p++) {
    const r = d[i];
    const g = d[i + 1];
    const b = d[i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    s.gray[p] = y > 255 ? 255 : (y + 0.5) | 0;
    const rn = r - (g > b ? g : b);
    s.red[p] = rn < 0 ? 0 : rn;
  }
  return { peak: brightestMean(s.gray, s.hist), peakRed: brightestMean(s.red, s.hist) };
}

// ---------------------------------------------------------------- capture

/** The two 160x120 planes of one camera frame, kept only when the rig is connected. */
interface FramePlanes {
  gray: Uint8Array;
  red: Uint8Array;
}

interface Capture {
  flips: FlipRecord[];
  rafTimestamps: number[];
  samples: CamSample[];
  /** Aligned 1:1 with `samples`, or null when no rig was connected. ~70 MB for 60 s. */
  planes: FramePlanes[] | null;
  rvfc: boolean;
}

type Schedule =
  { kind: "mseq"; seq: Uint8Array; symbolLen: number } | { kind: "sparse"; times: number[] };

/**
 * Runs the level schedule and the camera sampler concurrently for `durationS` seconds.
 * `panel` is already on screen at levels[0].
 */
function capture(
  panel: HTMLDivElement,
  schedule: Schedule,
  levels: [string, string],
  durationS: number,
  storePlanes: boolean,
): Promise<Capture> {
  const flips: FlipRecord[] = [];
  const rafTimestamps: number[] = [];
  const samples: CamSample[] = [];
  const planes: FramePlanes[] | null = storePlanes ? [] : null;

  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE_W;
  canvas.height = SAMPLE_H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d canvas context unavailable");
  const sampler = getPlaneSampler();

  const rvfc = asRvfc(video);
  let running = true;

  function grab(now: number, meta: VideoFrameMeta | null): void {
    ctx!.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
    const d = ctx!.getImageData(0, 0, SAMPLE_W, SAMPLE_H).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) {
      sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    }
    const { peak, peakRed } = samplePlanes(sampler);
    // The planes are scratch buffers, so keeping a frame means copying it.
    if (planes) planes.push({ gray: sampler.gray.slice(), red: sampler.red.slice() });
    samples.push({
      now,
      captureTime: meta?.captureTime ?? null,
      receiveTime: meta?.receiveTime ?? null,
      expectedDisplayTime: meta?.expectedDisplayTime ?? null,
      presentedFrames: meta?.presentedFrames ?? null,
      lum: sum / (SAMPLE_W * SAMPLE_H),
      peak,
      peakRed,
    });
  }

  return new Promise<Capture>((resolve) => {
    const durationMs = durationS * 1000;
    let rafCount = 0;
    let symbolIdx = 0;
    let level = -1;
    let t0 = -1;
    let flipHandle = 0;
    let camHandle = 0;

    function onKey(ev: KeyboardEvent): void {
      // Escape aborts early — a way out of the flashing without waiting it out.
      if (ev.key === "Escape") finish();
    }

    function finish(): void {
      if (!running) return;
      running = false;
      window.removeEventListener("keydown", onKey);
      cancelAnimationFrame(flipHandle);
      if (rvfc) rvfc.cancelVideoFrameCallback(camHandle);
      else cancelAnimationFrame(camHandle);
      resolve({ flips, rafTimestamps, samples, planes, rvfc: rvfc !== null });
    }

    // Flip driver. The rAF timestamp is our time reference for the flip; the photons for that
    // paint leave the panel some time later, and measuring exactly that delay is the point.
    function setLevel(next: number, ts: number): void {
      if (next === level) return;
      level = next;
      panel.style.background = levels[level];
      flips.push({ t: ts, tNow: performance.now(), level });
    }

    function flipFrame(ts: number): void {
      if (!running) return;
      if (t0 < 0) t0 = ts;
      rafTimestamps.push(ts);
      if (schedule.kind === "mseq") {
        if (rafCount % schedule.symbolLen === 0) {
          setLevel(schedule.seq[symbolIdx % schedule.seq.length], ts);
          symbolIdx++;
        }
      } else {
        // First frame pins level 0 at t0 so the stimulus is defined from the start;
        // then toggle at each scheduled time (relative to t0).
        if (rafCount === 0) setLevel(0, ts);
        while (symbolIdx < schedule.times.length && ts - t0 >= schedule.times[symbolIdx]) {
          setLevel(1 - level, ts);
          symbolIdx++;
        }
      }
      rafCount++;
      if (ts - t0 >= durationMs) {
        finish();
        return;
      }
      flipHandle = requestAnimationFrame(flipFrame);
    }

    // Camera sampler.
    function camFrame(now: number, meta: VideoFrameMeta): void {
      if (!running) return;
      grab(now, meta);
      camHandle = rvfc!.requestVideoFrameCallback(camFrame);
    }

    function camRaf(ts: number): void {
      if (!running) return;
      grab(ts, null);
      camHandle = requestAnimationFrame(camRaf);
    }

    window.addEventListener("keydown", onKey);

    if (rvfc) camHandle = rvfc.requestVideoFrameCallback(camFrame);
    else camHandle = requestAnimationFrame(camRaf);
    flipHandle = requestAnimationFrame(flipFrame);
  });
}

// ---------------------------------------------------------------- analysis

function pick(samples: CamSample[], key: "captureTime" | "receiveTime" | "now"): LumSample[] {
  const out: LumSample[] = [];
  for (const s of samples) {
    const t = key === "now" ? s.now : s[key];
    if (t === null || !Number.isFinite(t)) continue;
    out.push({ t, lum: s.lum });
  }
  return out;
}

function droppedFrames(samples: CamSample[]): number {
  let dropped = 0;
  let prev: number | null = null;
  for (const s of samples) {
    if (s.presentedFrames === null) continue;
    if (prev !== null) {
      const gap = s.presentedFrames - prev - 1;
      if (gap > 0) dropped += gap;
    }
    prev = s.presentedFrames;
  }
  return dropped;
}

function verdictFor(
  mode: Mode,
  primary: LagResult | null,
  halves: { first: LagResult; second: LagResult } | null,
  nEdges: number,
): string {
  if (!primary) return "UNRELIABLE — no usable camera samples";
  const reasons: string[] = [];
  // D(τ) is normalised so 1.0 = every edge's full jump captured; noise and partial-exposure
  // frames pull it down, but below MIN_PEAK_D most edges were not seen cleanly.
  const stat = mode === "sparse" ? "D" : "r";
  const floor = mode === "sparse" ? MIN_PEAK_D : MIN_PEAK_CORR;
  if (!(primary.peakCorr >= floor))
    reasons.push(`weak peak (${stat}=${num(primary.peakCorr, 3)} < ${floor})`);
  if (mode === "sparse") {
    // With sparse edges r decays slowly away from the peak (only samples near an edge are
    // misclassified), so the second-peak ratio is uninformative; what matters is that the
    // edges intersect to a window narrower than one camera frame.
    if (!(primary.plateau.widthMs <= MAX_PLATEAU_WIDTH_MS))
      reasons.push(
        `plateau ${num(primary.plateau.widthMs, 1)} ms wide (> ${MAX_PLATEAU_WIDTH_MS}; too few edges)`,
      );
  } else if (!(primary.secondPeakRatio <= MAX_SECOND_PEAK_RATIO)) {
    reasons.push(
      `ambiguous peak (second/peak=${num(primary.secondPeakRatio, 2)} > ${MAX_SECOND_PEAK_RATIO})`,
    );
  }
  if (halves) {
    const pick = (r: LagResult): number =>
      mode === "sparse" ? r.plateau.centerMs : r.lagRefinedMs;
    const gap = Math.abs(pick(halves.first) - pick(halves.second));
    // The 20 ms default is the estimator's own tail on ~5 s of data. A long sparse run puts
    // far more edges in each half, so the halves should agree much more tightly than that —
    // at 15+ edges each, a 20 ms disagreement is a real inconsistency, not sampling noise.
    const perHalf = Math.floor(nEdges / 2);
    const tol =
      mode === "sparse" && perHalf >= DENSE_HALF_MIN_EDGES
        ? MAX_HALF_DISAGREEMENT_DENSE_MS
        : MAX_HALF_DISAGREEMENT_MS;
    if (!(gap <= tol))
      reasons.push(`halves disagree by ${num(gap)} ms (> ${tol} ms at ${perHalf} edges/half)`);
  } else {
    reasons.push("no half-split check (captureTime unavailable)");
  }
  return reasons.length === 0 ? "OK" : `UNRELIABLE — ${reasons.join("; ")}`;
}

// ---------------------------------------------------------------- rendering

function renderReport(settings: Settings, results: Results, cap: Capture): string {
  const lines: string[] = [];
  const sparse = settings.mode === "sparse";
  lines.push(
    `clock        ${padL("estimate", 9)} ${padL("plateau", 14)} ${padL("argmax", 8)} ${padL(sparse ? "peak D" : "peak r", 8)} ${padL("2nd/peak", 9)} ${padL("n", 6)}`,
  );
  lines.push("-".repeat(72));
  for (const v of results.variants) {
    const plateau = `${num(v.plateauCenterMs, 1)} ±${num(v.plateauWidthMs / 2, 1)}`;
    lines.push(
      `${pad(v.clock, 12)} ${padL(num(estimateOf(v, settings.mode), 1), 9)} ${padL(plateau, 14)} ` +
        `${padL(num(v.lagRefinedMs, 1), 8)} ${padL(num(v.peakCorr, 3), 8)} ` +
        `${padL(sparse ? "n/a" : num(v.secondPeakRatio, 3), 9)} ${padL(String(v.nSamples), 6)}`,
    );
  }
  lines.push(
    sparse
      ? "             edge-difference statistic D(τ) (1.0 = every edge's full jump captured);" +
          " estimate = plateau center, ± = half its width"
      : "             step-template correlation; estimate = parabolic-refined argmax",
  );
  lines.push("");
  if (results.halves) {
    const { first, second } = results.halves;
    const a = estimateOf(first, settings.mode);
    const b = estimateOf(second, settings.mode);
    const gap = Math.abs(a - b);
    lines.push(
      `half split   first ${num(a)} ms (r=${num(first.peakCorr, 3)}, n=${first.nSamples})  ` +
        `second ${num(b)} ms (r=${num(second.peakCorr, 3)}, n=${second.nSamples})  Δ=${num(gap)} ms`,
    );
  } else {
    lines.push("half split   n/a (no captureTime)");
  }
  lines.push("");
  lines.push(`camera dt    ${statsLine(results.cameraInterval)}`);
  lines.push(`dropped cam  ${results.droppedCameraFrames} frame(s)`);
  lines.push(`rAF dt       ${statsLine(results.rafInterval)}`);
  lines.push(`flips        ${results.nFlips}   rAF callbacks ${cap.rafTimestamps.length}`);
  const modeDesc = sparse
    ? `sparse steps, gaps ${settings.gapMinMs}–${settings.gapMaxMs} ms, seed ${settings.seed}`
    : `m-sequence order ${settings.order} (${settings.sequenceLength} symbols), symbolLen ${settings.symbolLen} ` +
      `refreshes ≈ ${settings.symbolMsNominal === null ? "n/a" : num(settings.symbolMsNominal, 1) + " ms"}`;
  lines.push(
    `settings     ${modeDesc}  contrast ${settings.contrast}  duration ${settings.durationS}s  ` +
      `video ${settings.videoWidth}×${settings.videoHeight}@${settings.frameRate ?? "?"}  ` +
      `exposure ${settings.exposureMode ?? "?"}  rVFC ${settings.rvfc ? "yes" : "no (rAF fallback)"}`,
  );
  lines.push("");
  lines.push(`verdict      ${results.verdict}`);
  return lines.join("\n");
}

function drawCurve(r: LagResult | null, mode: Mode = "sparse"): void {
  const ctx = curveEl.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, CURVE_W, CURVE_H);
  ctx.fillStyle = "#0c0c0c";
  ctx.fillRect(0, 0, CURVE_W, CURVE_H);
  if (!r || r.curve.length < 2) {
    ctx.fillStyle = "#666";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("no curve", 8, 20);
    return;
  }
  const padX = 34;
  const padY = 12;
  const x0 = r.curve[0].lagMs;
  const x1 = r.curve[r.curve.length - 1].lagMs;
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of r.curve) {
    if (p.r < lo) lo = p.r;
    if (p.r > hi) hi = p.r;
  }
  lo = Math.min(lo, 0);
  hi = Math.max(hi, 0.001);
  const sx = (v: number): number => padX + ((v - x0) / (x1 - x0 || 1)) * (CURVE_W - padX - 8);
  const sy = (v: number): number =>
    CURVE_H - padY - ((v - lo) / (hi - lo || 1)) * (CURVE_H - 2 * padY);

  // zero line
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, sy(0));
  ctx.lineTo(CURVE_W - 8, sy(0));
  ctx.stroke();

  // plateau extent, then the headline estimate
  const est = mode === "sparse" ? r.plateau.centerMs : r.lagRefinedMs;
  if (Number.isFinite(r.plateau.lowMs)) {
    ctx.fillStyle = "rgba(200, 136, 85, 0.25)";
    const xa = sx(r.plateau.lowMs);
    const xb = Math.max(sx(r.plateau.highMs), xa + 1);
    ctx.fillRect(xa, padY, xb - xa, CURVE_H - 2 * padY);
  }
  ctx.strokeStyle = "#c85";
  ctx.beginPath();
  ctx.moveTo(sx(est), padY);
  ctx.lineTo(sx(est), CURVE_H - padY);
  ctx.stroke();

  // curve
  ctx.strokeStyle = "#6cf";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  r.curve.forEach((p, i) => {
    const x = sx(p.lagMs);
    const y = sy(p.r);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#888";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(hi.toFixed(2), 2, sy(hi) + 4);
  ctx.fillText(lo.toFixed(2), 2, sy(lo) + 4);
  ctx.fillText(`${x0} ms`, padX, CURVE_H - 2);
  ctx.fillText(`${x1} ms`, CURVE_W - 52, CURVE_H - 2);
  ctx.fillStyle = "#c85";
  ctx.fillText(`lag ${est.toFixed(1)} ms`, Math.min(sx(est) + 4, CURVE_W - 96), padY + 10);
}

// ---------------------------------------------------------------- hardware rig
//
// Optional: a photodiode on the screen and an LED in the camera's view, on a microcontroller
// with its own microsecond clock (web/rig). It rides along with an ordinary loopback run and
// answers what the loopback cannot — the display-side/camera-side split, and whether the one
// number the loopback reports actually closes the loop. All the maths is in ./rigAnalysis.

interface RigPanelSettings {
  relayMode: 0 | 1 | 2;
  delayMinUs: number;
  delayMaxUs: number;
  pulseWidthUs: number;
  train: boolean;
  gapMinMs: number;
  gapMaxMs: number;
  reportEdges: boolean;
}

interface RigRun {
  settings: RigPanelSettings;
  info: RigInfo | null;
  eventStart: number;
  syncBefore: SyncPair[];
  syncAfter: SyncPair[];
  events: RigEvent[];
  /** The run's 160x120 planes, aligned with its camera samples. Never exported. */
  planes: FramePlanes[] | null;
}

/** Everything about the LED signal the analysis ran on, for the report and the export. */
interface RigRoiSummary {
  source: LedSource;
  plane: LedPlane | null;
  centroid: { x: number; y: number } | null;
  k: number;
  n: number;
  contrast: number;
  contrastKth: number;
  /** ROI-signal separation in counts, per plane (NaN when that plane did not localise). */
  contrastGray: number;
  contrastRed: number;
  /** The same in units of the baseline MAD — what the two planes are compared on. */
  madsGray: number;
  madsRed: number;
  nOn: number;
  nOff: number;
  indices: number[];
  /** The ROI the aiming panel found, and how far its centroid is from this one. */
  panelCentroid: { x: number; y: number } | null;
  panelDistancePx: number;
}

interface RigReport {
  settings: RigPanelSettings;
  info: RigInfo | null;
  syncPairs: SyncPair[];
  fit: SyncFit;
  events: RigEvent[];
  nEdges: number;
  nLeds: number;
  /** The loopback's own estimate, the number under test. */
  loopbackLagMs: number;
  roi: RigRoiSummary;
  display: DisplayLagResult;
  relay: RelayResult;
  exposure: ExposureResult;
  rowBands: RowBandResult | null;
  /**
   * plateau + half the exposure + (mean − min display lag): the constant to subtract from a
   * gaze sample's captureTime to land in the middle of the camera's integration for a flip
   * with average display lag, rather than at the frame boundary the plateau measures.
   */
  midExposureIdealMs: number;
  /** relay.plateau.centerMs − loopbackLagMs: the number the verdict is about. */
  plateauDeltaMs: number;
  verdict: string;
}

const rig = new Rig();
let rigEdgeCount = 0;
let rigLedCount = 0;

const rigConnectBtn = el<HTMLButtonElement>("rig-connect");
const rigDisconnectBtn = el<HTMLButtonElement>("rig-disconnect");
const rigSyncBtn = el<HTMLButtonElement>("rig-sync");
const rigPulseBtn = el<HTMLButtonElement>("rig-pulse");
const rigStatusEl = el<HTMLSpanElement>("rig-status");
const rigCountsEl = el<HTMLSpanElement>("rig-counts");
const rigOutEl = el<HTMLPreElement>("rigout");
const rigPlotEl = el<HTMLCanvasElement>("rigplot");

function setRigStatus(s: string): void {
  rigStatusEl.textContent = s;
}

function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function readRigSettings(): RigPanelSettings {
  const modeRaw = Number(el<HTMLSelectElement>("rig-mode").value);
  const relayMode: 0 | 1 | 2 = modeRaw === 1 ? 1 : modeRaw === 2 ? 2 : 0;
  const delayMinUs = Math.max(0, Number(el<HTMLInputElement>("rig-delay-min").value) || 100000);
  const delayMaxUs = Math.max(
    delayMinUs,
    Number(el<HTMLInputElement>("rig-delay-max").value) || 140000,
  );
  const gapMinMs = Math.max(1, Number(el<HTMLInputElement>("rig-gap-min").value) || 600);
  const gapMaxMs = Math.max(gapMinMs, Number(el<HTMLInputElement>("rig-gap-max").value) || 1200);
  return {
    relayMode,
    delayMinUs,
    delayMaxUs,
    pulseWidthUs: Math.max(100, Number(el<HTMLInputElement>("rig-pulse-width").value) || 100000),
    train: el<HTMLInputElement>("rig-train").checked,
    gapMinMs,
    gapMaxMs,
    reportEdges: el<HTMLInputElement>("rig-edges").checked,
  };
}

async function applyRigSettings(s: RigPanelSettings): Promise<void> {
  await rig.setEdgeReporting(s.reportEdges);
  await rig.setPulseWidth(s.pulseWidthUs);
  await rig.setDelayRange(s.delayMinUs, s.delayMaxUs);
  await rig.setGaps(s.gapMinMs, s.gapMaxMs);
  await rig.setTrain(s.train);
  await rig.setRelay(s.relayMode);
}

/**
 * What the user needs to see about a connected rig: which port, and whether the photodiode
 * is seeing anything. The envelope only separates once the screen has shown both levels, so
 * it reads narrow until the first run — that is expected, not a fault.
 */
function describeInfo(info: RigInfo): string {
  const env =
    typeof info.env_min === "number" && typeof info.env_max === "number"
      ? `photodiode ${info.env_min}–${info.env_max} (swing ${info.env_max - info.env_min})`
      : "photodiode ?";
  const level = `level ${String(info.pd_level ?? "?")}`;
  const dropped =
    typeof info.dropped_lines === "number" ? `, ${info.dropped_lines} dropped line(s)` : "";
  return `${env}, ${level}${dropped}`;
}

function setConnectedStatus(info: RigInfo | null): void {
  setRigStatus(
    info
      ? `connected on ${rig.portLabel()} — ${describeInfo(info)}`
      : `connected on ${rig.portLabel()}`,
  );
}

function updateRigCounts(): void {
  rigCountsEl.textContent = `edges ${rigEdgeCount} · LEDs ${rigLedCount}`;
}

rig.onEvent = (e: RigEvent): void => {
  if (e.kind === "E") rigEdgeCount++;
  else if (e.kind === "L") rigLedCount++;
  else return;
  updateRigCounts();
};

function toRigEdges(events: RigEvent[]): RigEdge[] {
  return events
    .filter((e): e is RigEdgeEvent => e.kind === "E")
    .map((e) => ({ tUs: e.tUs, dir: e.dir, level: e.level, envMin: e.envMin, envMax: e.envMax }));
}

function toRigLeds(events: RigEvent[]): RigLed[] {
  return events
    .filter((e): e is RigLedEvent => e.kind === "L")
    .map((e) => ({
      tOnUs: e.tOnUs,
      tOffUs: e.tOffUs,
      src: e.src,
      delayUs: e.delayUs,
      edgeTUs: e.edgeTUs,
    }));
}

function toCamFrames(samples: CamSample[], plane: LedPlane = "gray"): CamFrame[] {
  const out: CamFrame[] = [];
  for (const s of samples) {
    if (s.captureTime === null || !Number.isFinite(s.captureTime)) continue;
    out.push({
      captureTime: s.captureTime,
      now: s.now,
      peak: plane === "red" ? s.peakRed : s.peak,
    });
  }
  return out;
}

/** Frames with their 160x120 planes attached, for rig-informed LED localisation. */
function toCamFrameGrays(samples: CamSample[], planes: FramePlanes[]): CamFrameGray[] {
  const out: CamFrameGray[] = [];
  for (let i = 0; i < samples.length && i < planes.length; i++) {
    const s = samples[i];
    if (s.captureTime === null || !Number.isFinite(s.captureTime)) continue;
    out.push({
      captureTime: s.captureTime,
      now: s.now,
      peak: s.peak,
      peakRed: s.peakRed,
      gray: planes[i].gray,
      red: planes[i].red,
    });
  }
  return out;
}

function summariseRoi(
  source: LedSource,
  roi: RoiResult | null,
  roiGray: RoiResult | null,
  roiRed: RoiResult | null,
  panel: RoiResult | null,
): RigRoiSummary {
  const centroid = roi ? roi.centroid : null;
  const panelCentroid = panel ? panel.centroid : null;
  return {
    source,
    plane: roi ? roi.plane : null,
    centroid,
    k: roi ? roi.k : 0,
    n: roi ? roi.n : 0,
    contrast: roi ? roi.contrast : NaN,
    contrastKth: roi ? roi.contrastKth : NaN,
    contrastGray: roiGray ? roiGray.signalContrast : NaN,
    contrastRed: roiRed ? roiRed.signalContrast : NaN,
    madsGray: roiGray ? roiGray.signalMads : NaN,
    madsRed: roiRed ? roiRed.signalMads : NaN,
    nOn: roi ? roi.nOn : 0,
    nOff: roi ? roi.nOff : 0,
    indices: roi ? roi.indices : [],
    panelCentroid,
    panelDistancePx:
      centroid && panelCentroid
        ? Math.hypot(centroid.x - panelCentroid.x, centroid.y - panelCentroid.y)
        : NaN,
  };
}

/**
 * The verdict is about the plateau, not the mean.
 *
 * A first-lit frame is stamped somewhere in the frame *after* the LED came on, so the mean
 * of those stamps sits about half a frame period above the true lag no matter how right the
 * correction is — testing it against zero would fail a perfect setup. The plateau is the
 * intersection of the per-pulse brackets, which is exactly what the loopback's edge
 * estimator computes from screen edges, so their difference is meaningful.
 */
function rigVerdict(relay: RelayResult, loopbackLagMs: number): string {
  if (relay.ledNotDetected)
    return "UNRELIABLE — the LED never separated from the background; check aim, brightness and diffuser";
  if (relay.n < 5)
    return `UNRELIABLE — only ${relay.n} relayed pulse(s) matched a flip and a frame`;
  const p = relay.plateau;
  if (!Number.isFinite(p.centerMs))
    return "UNRELIABLE — no lag is consistent with any pulse (check the sync fit and the flip pairing)";
  const delta = p.centerMs - loopbackLagMs;
  const frac = p.n > 0 ? p.consistent / p.n : 0;
  const reasons: string[] = [];
  if (!(Math.abs(delta) <= MAX_RIG_RESIDUAL_MS))
    reasons.push(
      `the LED plateau is ${signed(delta)} ms from the loopback estimate ` +
        `(> ${MAX_RIG_RESIDUAL_MS} ms: the correction is off by about that much)`,
    );
  if (!(frac >= MIN_RIG_CONSISTENT_FRACTION))
    reasons.push(
      `only ${p.consistent}/${p.n} pulses agree on a single lag ` +
        `(< ${Math.round(MIN_RIG_CONSISTENT_FRACTION * 100)} %: display-lag jitter, or mis-paired pulses)`,
    );
  return reasons.length === 0 ? "OK" : `UNRELIABLE — ${reasons.join("; ")}`;
}

function analyseRig(
  run: RigRun,
  flips: Flip[],
  samples: CamSample[],
  loopbackLagMs: number,
): RigReport {
  const edges = toRigEdges(run.events);
  const leds = toRigLeds(run.events);
  const syncPairs = [...run.syncBefore, ...run.syncAfter];
  const fit = fitSync(syncPairs);

  // The LED signal: where the rig's own record of when the LED was lit says the LED is,
  // rather than wherever the brightest pixels happen to be. Falls back to the brightest-N
  // redness mean when the pulses could not localise it.
  const choice = run.planes
    ? chooseLedSignal(toCamFrameGrays(samples, run.planes), leds, fit, PEAK_W, PEAK_H)
    : {
        source: "peak24-red" as LedSource,
        frames: toCamFrames(samples, "red"),
        roi: null,
        roiGray: null,
        roiRed: null,
      };
  const frames = choice.frames;
  const relay = relayResiduals(leds, edges, flips, frames, fit, loopbackLagMs);
  const display = displayLag(flips, edges, fit);

  // Where in the frame the lag is measured matters if the sensor has a rolling shutter, so
  // measure it in four horizontal bands and see whether it drifts down the frame.
  const rows = run.planes
    ? rowBands(
        toCamFrameGrays(samples, run.planes),
        flips,
        PEAK_W,
        PEAK_H,
        loopbackLagMs,
        choice.roi ? choice.roi.centroid.y : null,
      )
    : null;

  // The plateau lands on the frame boundary and on the *shortest* display lag (the pulses
  // with a longer one are what shrink the intersection from above), so mid-exposure
  // alignment for a typical flip adds half the exposure and the display-lag offset.
  const allLags = allLagStats(display);
  const midExposureIdealMs =
    relay.plateau.centerMs + relay.exposureEstimateMs / 2 + (allLags.meanMs - allLags.minMs);

  return {
    settings: run.settings,
    info: run.info,
    syncPairs,
    fit,
    events: run.events,
    nEdges: edges.length,
    nLeds: leds.length,
    loopbackLagMs,
    roi: summariseRoi(choice.source, choice.roi, choice.roiGray, choice.roiRed, panelRoi),
    display,
    relay,
    exposure: exposureWindow(leds, frames, fit),
    rowBands: rows,
    midExposureIdealMs,
    plateauDeltaMs: relay.plateau.centerMs - loopbackLagMs,
    verdict: rigVerdict(relay, loopbackLagMs),
  };
}

/** Display lag over both directions together — the rig fires off whichever edge it is set to. */
function allLagStats(d: DisplayLagResult): LagStats {
  const lags = d.pairs.map((p) => p.lagMs);
  const n = lags.length;
  if (n === 0) return { n: 0, meanMs: NaN, sdMs: NaN, minMs: NaN, maxMs: NaN };
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of lags) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / n;
  let ss = 0;
  for (const v of lags) ss += (v - mean) * (v - mean);
  return { n, meanMs: mean, sdMs: n > 1 ? Math.sqrt(ss / (n - 1)) : 0, minMs: min, maxMs: max };
}

function lagLine(label: string, s: LagStats): string {
  if (s.n === 0) return `             ${pad(label, 8)} n 0`;
  return (
    `             ${pad(label, 8)} n ${padL(String(s.n), 3)}  ${num(s.meanMs, 1)} ± ${num(s.sdMs, 1)} ms  ` +
    `[${num(s.minMs, 1)}, ${num(s.maxMs, 1)}]`
  );
}

function renderRigReport(r: RigReport): string {
  const lines: string[] = [];
  const ppm = Number.isFinite(r.fit.b) ? (r.fit.b - 1) * 1e6 : NaN;
  lines.push(
    `sync         n ${r.fit.n}/${r.syncPairs.length}  slope ${num(ppm, 0)} ppm  ` +
      `RTT median ${num(r.fit.rttMedianMs, 2)} / max ${num(r.fit.rttMaxMs, 2)} ms  ` +
      `residual sd ${num(r.fit.residualSdMs, 2)} ms`,
  );
  lines.push(`display lag  rig edges ${r.nEdges}, LED pulses ${r.nLeds}`);
  lines.push(lagLine("rising", r.display.rising));
  lines.push(lagLine("falling", r.display.falling));
  lines.push(
    `             unmatched: ${r.display.unmatchedFlips} flip(s), ${r.display.unmatchedEdges} edge(s)`,
  );
  lines.push("");
  const relay = r.relay;
  if (relay.n === 0) {
    lines.push(
      `relay        no residuals (${relay.unmatched} pulse(s) unmatched` +
        `${relay.ledNotDetected ? ", LED not detected" : ""})`,
    );
  } else {
    const p = relay.plateau;
    lines.push(
      `relay        plateau ${num(p.centerMs, 1)} ms [${num(p.lowMs, 1)}, ${num(p.highMs, 1)}] ` +
        `w ${num(p.widthMs, 1)}  ${p.consistent}/${p.n} consistent  ` +
        `Δ vs loopback ${signed(r.plateauDeltaMs)} ms`,
    );
    lines.push(
      `             n ${relay.n} (${relay.unmatched} unmatched)  ` +
        `mean first-lit stamp ${num(relay.meanDirectLagMs, 1)} ± ${num(relay.sdDirectLagMs, 1)} ms ` +
        `(residual vs L ${num(relay.meanResidualMs, 1)})`,
    );
    lines.push(
      "             that mean sits ~half a frame after the event by construction — not a bias test;",
    );
    lines.push("             the plateau above is the number comparable with the loopback.");
    lines.push(
      `             exposure ≈ ${num(relay.exposureEstimateMs, 1)} ms ` +
        `(${num(100 * relay.fullJumpFraction, 0)} % of pulses jumped full ${num(relay.fullJump, 1)}, ` +
        `frame ${num(relay.framePeriodMs, 1)} ms)`,
    );
    lines.push(
      `             mid-exposure ideal ≈ ${num(r.midExposureIdealMs, 1)} ms = plateau + E/2 + ` +
        `(mean − min display lag)`,
    );
    lines.push(
      `             ${padL("delay ms", 10)} ${padL("residual ms", 12)} ${padL("f", 6)} ${padL("jump", 8)}`,
    );
    for (const row of relay.residuals.slice(0, MAX_RESIDUAL_ROWS)) {
      lines.push(
        `             ${padL(num(row.delayMs, 1), 10)} ${padL(num(row.residualMs, 1), 12)} ` +
          `${padL(num(row.f, 2), 6)} ${padL(num(row.jump, 1), 8)}`,
      );
    }
    if (relay.residuals.length > MAX_RESIDUAL_ROWS)
      lines.push(`             … ${relay.residuals.length - MAX_RESIDUAL_ROWS} more (in the JSON)`);
  }
  lines.push("");
  const exp = r.exposure;
  lines.push(
    exp.n === 0
      ? "exposure     no train pulses"
      : `exposure     n ${exp.n}  window [${num(exp.startMs, 1)}, ${num(exp.endMs, 1)}] ms around captureTime, ` +
          `width ${num(exp.windowMs, 1)} ms`,
  );
  lines.push("");
  const rows = r.rowBands;
  if (rows && rows.bands.length > 0) {
    const dir =
      rows.direction === "top-first"
        ? "top→bottom"
        : rows.direction === "bottom-first"
          ? "bottom→top"
          : "flat";
    lines.push(
      `rows         readout ≈ ${num(rows.readoutMs, 1)} ms ${dir}  ` +
        `(${num(rows.slopeMsPerRow * 1000, 2)} µs/row over ${PEAK_H} rows)`,
    );
    lines.push(
      `             bands ${rows.bands.map((b) => `${num(b.lagMs, 1)}@r${num(b.row, 0)}`).join(" / ")}`,
    );
    lines.push(
      `             LED row ${rows.ledRow === null ? "n/a" : num(rows.ledRow, 0)} → ${num(rows.ledRowLagMs, 1)} ms; ` +
        `whole frame ${num(rows.wholeFrameLagMs, 1)} ms; offset ${signed(rows.offsetMs)} ms`,
    );
    lines.push("");
  }
  const roi = r.roi;
  const planeContrasts =
    `red ${num(roi.contrastRed, 1)} counts (${num(roi.madsRed, 1)} MAD) · ` +
    `gray ${num(roi.contrastGray, 1)} counts (${num(roi.madsGray, 1)} MAD)`;
  if (roi.centroid) {
    lines.push(
      `led signal   ${roi.source}: ${roi.n}/${roi.k} px at ` +
        `(${num(roi.centroid.x, 1)}, ${num(roi.centroid.y, 1)}) of ${PEAK_W}×${PEAK_H}, ` +
        `on/off frames ${roi.nOn}/${roi.nOff}`,
    );
    lines.push(`             ${planeContrasts}`);
    lines.push(
      `             weakest ROI px ${num(roi.contrast, 1)} counts, ${roi.k}th-ranked ${num(roi.contrastKth, 1)}`,
    );
    if (roi.panelCentroid)
      lines.push(
        `             panel ROI (${num(roi.panelCentroid.x, 1)}, ${num(roi.panelCentroid.y, 1)}) — ` +
          `${num(roi.panelDistancePx, 1)} px away` +
          (roi.panelDistancePx <= ROI_AGREE_PX ? " (agrees)" : " (DISAGREES — was the LED moved?)"),
      );
  } else {
    lines.push(
      `led signal   ${roi.source} — rig-informed localisation failed (too few pulses, or no LED in frame);` +
        ` falling back to the brightest ${PEAK_N} redness pixels`,
    );
  }
  const det = relay.detector;
  lines.push(
    `led detector baseline ${num(det.baseline, 1)}  led ${num(det.ledLevel, 1)}  ` +
      `mad ${num(det.madBaseline, 2)}  threshold ${num(det.threshold, 1)}  ` +
      `frames in/out ${det.nInside}/${det.nOutside}`,
  );
  lines.push("");
  lines.push(`verdict      ${r.verdict}`);
  return lines.join("\n");
}

/** Left: relay residual vs the delay the rig applied. Right: exposure f vs phase. */
function drawRigPlot(r: RigReport | null): void {
  const ctx = rigPlotEl.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, RIG_W, RIG_H);
  ctx.fillStyle = "#0c0c0c";
  ctx.fillRect(0, 0, RIG_W, RIG_H);
  ctx.font = "10px ui-monospace, monospace";
  const half = RIG_W / 2;

  function panel(
    x0: number,
    title: string,
    pts: { x: number; y: number }[],
    yLo: number,
    yHi: number,
    zeroLine: boolean,
  ): void {
    const padX = 30;
    const padY = 14;
    const w = half - padX - 10;
    const h = RIG_H - 2 * padY;
    ctx!.fillStyle = "#888";
    ctx!.fillText(title, x0 + padX, 10);
    if (pts.length === 0) {
      ctx!.fillStyle = "#555";
      ctx!.fillText("no data", x0 + padX, RIG_H / 2);
      return;
    }
    let xLo = Infinity;
    let xHi = -Infinity;
    for (const p of pts) {
      if (p.x < xLo) xLo = p.x;
      if (p.x > xHi) xHi = p.x;
    }
    if (!(xHi > xLo)) xHi = xLo + 1;
    const sx = (v: number): number => x0 + padX + ((v - xLo) / (xHi - xLo)) * w;
    const sy = (v: number): number => RIG_H - padY - ((v - yLo) / (yHi - yLo || 1)) * h;
    ctx!.strokeStyle = "#333";
    ctx!.lineWidth = 1;
    ctx!.beginPath();
    const yRef = zeroLine ? 0 : yLo;
    ctx!.moveTo(x0 + padX, sy(yRef));
    ctx!.lineTo(x0 + padX + w, sy(yRef));
    ctx!.stroke();
    ctx!.fillStyle = "#6cf";
    for (const p of pts) {
      ctx!.fillRect(sx(p.x) - 1.5, sy(p.y) - 1.5, 3, 3);
    }
    ctx!.fillStyle = "#666";
    ctx!.fillText(num(yHi, 1), x0 + 2, sy(yHi) + 4);
    ctx!.fillText(num(yLo, 1), x0 + 2, sy(yLo) + 4);
    ctx!.fillText(num(xLo, 0), x0 + padX, RIG_H - 2);
    ctx!.fillText(num(xHi, 0), x0 + half - 40, RIG_H - 2);
  }

  const residuals = r ? r.relay.residuals.map((p) => ({ x: p.delayMs, y: p.residualMs })) : [];
  let span = 10;
  for (const p of residuals) span = Math.max(span, Math.abs(p.y) * 1.1);
  panel(0, "residual (ms) vs delay (ms)", residuals, -span, span, true);
  const exposure = r ? r.exposure.points.map((p) => ({ x: p.phaseMs, y: p.f })) : [];
  panel(half, "f vs phase (ms)", exposure, 0, 1, false);
}

async function connectRig(): Promise<void> {
  rigConnectBtn.disabled = true;
  try {
    await rig.connect();
    setRigStatus("connected — asking for info…");
    const info = await rig.info();
    setConnectedStatus(info);
    rigDisconnectBtn.disabled = false;
    rigSyncBtn.disabled = false;
    rigPulseBtn.disabled = false;
    syncRigUi();
  } catch (err) {
    setRigStatus(`connect failed: ${errText(err)}`);
    rigConnectBtn.disabled = false;
  }
}

async function disconnectRig(): Promise<void> {
  rigDisconnectBtn.disabled = true;
  rigSyncBtn.disabled = true;
  rigPulseBtn.disabled = true;
  try {
    if (rig.connected) {
      await rig.setRelay(0);
      await rig.setTrain(false);
    }
  } catch {
    /* the device may already be gone; close anyway */
  }
  await rig.disconnect();
  setRigStatus("disconnected");
  rigConnectBtn.disabled = false;
  syncRigUi();
}

// ---------------------------------------------------------------- aiming preview
//
// Aiming a 5 mm LED into a webcam's frame blind is hopeless, so the panel can run the camera
// on its own: a mirrored preview, the two brightest-N numbers live, a difference view that
// makes anything blinking pop out of a static scene, and a "locate" button that blinks the
// LED deliberately and marks where it found it.

const PREVIEW_W = 320;
const PREVIEW_H = 240;
/** Background EMA of the difference view: ~2 s at 30 fps. */
const DIFF_BG_ALPHA = 1 / 60;
/** Never stretch a difference smaller than this to full scale — that is just noise. */
const DIFF_MIN_SPAN = 6;
const STATS_INTERVAL_MS = 250;
const LOCATE_PULSES = 3;
const LOCATE_WIDTH_US = 400000;
const LOCATE_GAP_MS = 700;
const LOCATE_SYNC_PINGS = 10;
/** Panel and post-run ROI centroids further apart than this are reported as disagreeing. */
const ROI_AGREE_PX = 4;

const previewBtn = el<HTMLButtonElement>("rig-preview");
const locateBtn = el<HTMLButtonElement>("rig-locate");
const aimRow = el("aim-row");
const diffSel = el<HTMLSelectElement>("rig-diff");
const previewCanvas = el<HTMLCanvasElement>("rig-preview-canvas");
const previewStatsEl = el<HTMLSpanElement>("rig-preview-stats");
const roiEl = el<HTMLSpanElement>("rig-roi");

let previewStream: MediaStream | null = null;
let previewOn = false;
let previewHandle = 0;
let previewRvfc: RvfcVideoElement | null = null;
let diffBg: Float32Array | null = null;
let diffBuf: Float32Array | null = null;
let lastStatsAt = 0;
/** Non-null while "locate" is collecting frames from the preview loop. */
let locateFrames: CamFrameGray[] | null = null;
/** The ROI the panel found, used for the live readout and compared against the run's own. */
let panelRoi: RoiResult | null = null;

/** One place for every "is this control usable right now" decision. */
function syncRigUi(): void {
  const connected = rig.connected;
  aimRow.hidden = !connected;
  previewBtn.disabled = runInProgress;
  locateBtn.disabled = !(connected && previewOn) || runInProgress;
  for (const p of PRESETS) p.button.disabled = runInProgress || (p.needsRig && !connected);
}

function drawRoiBox(ctx: CanvasRenderingContext2D, roi: RoiResult): void {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const i of roi.indices) {
    const x = i % roi.width;
    const y = Math.floor(i / roi.width);
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  if (!Number.isFinite(x0)) return;
  const sx = PREVIEW_W / roi.width;
  const sy = PREVIEW_H / roi.height;
  // The canvas is mirrored in CSS, so drawing in plain frame coordinates puts the box over
  // the LED where the user sees it.
  ctx.strokeStyle = "#6f6";
  ctx.lineWidth = 2;
  ctx.strokeRect(x0 * sx - 4, y0 * sy - 4, (x1 - x0 + 1) * sx + 8, (y1 - y0 + 1) * sy + 8);
}

/**
 * Basic mode always shows the redness difference: it is the view that makes a small red LED
 * findable, and there is nothing to choose. Advanced exposes the selector.
 */
function currentDiffMode(): string {
  return advanced ? diffSel.value : "red";
}

function drawPreview(s: PlaneSampler): void {
  const ctx = previewCanvas.getContext("2d");
  if (!ctx) return;
  const mode = currentDiffMode();
  if (mode === "gray" || mode === "red") {
    const plane = mode === "red" ? s.red : s.gray;
    if (!diffBg || diffBg.length !== plane.length) {
      diffBg = new Float32Array(plane.length);
      for (let i = 0; i < plane.length; i++) diffBg[i] = plane[i];
    }
    if (!diffBuf || diffBuf.length !== plane.length) diffBuf = new Float32Array(plane.length);
    let max = 0;
    for (let i = 0; i < plane.length; i++) {
      diffBg[i] += (plane[i] - diffBg[i]) * DIFF_BG_ALPHA;
      const d = Math.abs(plane[i] - diffBg[i]);
      diffBuf[i] = d;
      if (d > max) max = d;
    }
    const scale = 255 / Math.max(max, DIFF_MIN_SPAN);
    const px = s.image.data;
    for (let i = 0, j = 0; i < plane.length; i++, j += 4) {
      const v = diffBuf[i] * scale;
      const b = v > 255 ? 255 : v | 0;
      px[j] = b;
      px[j + 1] = b;
      px[j + 2] = b;
      px[j + 3] = 255;
    }
    s.ctx.putImageData(s.image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(s.canvas, 0, 0, PREVIEW_W, PREVIEW_H);
  } else {
    diffBg = null;
    ctx.drawImage(video, 0, 0, PREVIEW_W, PREVIEW_H);
  }
  if (panelRoi) drawRoiBox(ctx, panelRoi);
}

function scheduleNextPreview(): void {
  if (!previewOn) return;
  if (previewRvfc) previewHandle = previewRvfc.requestVideoFrameCallback(previewFrame);
  else
    previewHandle = requestAnimationFrame((ts) => {
      previewFrame(ts, null);
    });
}

function previewFrame(now: number, meta: VideoFrameMeta | null): void {
  if (!previewOn) return;
  const s = getPlaneSampler();
  const { peak, peakRed } = samplePlanes(s);
  if (locateFrames)
    locateFrames.push({
      // The planes are scratch buffers; a kept frame needs its own copy.
      captureTime: meta?.captureTime ?? now,
      now,
      peak,
      peakRed,
      gray: s.gray.slice(),
      red: s.red.slice(),
    });
  drawPreview(s);
  if (now - lastStatsAt >= STATS_INTERVAL_MS) {
    lastStatsAt = now;
    const roiText = panelRoi
      ? ` · ROI ${num(roiMean(panelRoi.plane === "red" ? s.red : s.gray, panelRoi), 1)} (${panelRoi.plane})`
      : "";
    previewStatsEl.textContent = `gray peak ${num(peak, 1)} · red peak ${num(peakRed, 1)}${roiText}`;
  }
  scheduleNextPreview();
}

function clearPreviewCanvas(): void {
  const ctx = previewCanvas.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#0c0c0c";
  ctx.fillRect(0, 0, PREVIEW_W, PREVIEW_H);
}

function stopPreview(): void {
  if (!previewOn) return;
  previewOn = false;
  if (previewRvfc) previewRvfc.cancelVideoFrameCallback(previewHandle);
  else cancelAnimationFrame(previewHandle);
  previewRvfc = null;
  if (previewStream) {
    for (const track of previewStream.getTracks()) track.stop();
    previewStream = null;
  }
  video.srcObject = null;
  diffBg = null;
  previewBtn.textContent = "Preview";
  previewStatsEl.textContent = "preview off";
  clearPreviewCanvas();
  syncRigUi();
}

async function startPreview(): Promise<void> {
  if (previewOn) return;
  previewBtn.disabled = true;
  try {
    previewStream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);
    video.srcObject = previewStream;
    await video.play();
    previewOn = true;
    previewRvfc = asRvfc(video);
    lastStatsAt = 0;
    previewBtn.textContent = "Stop preview";
    previewStatsEl.textContent = "…";
    scheduleNextPreview();
  } catch (err) {
    previewStatsEl.textContent = `camera unavailable: ${errText(err)}`;
    if (previewStream) {
      for (const track of previewStream.getTracks()) track.stop();
      previewStream = null;
    }
  }
  previewBtn.disabled = false;
  syncRigUi();
}

/**
 * Blink the LED deliberately and find it: three long pulses, the frames in between, and the
 * same on-minus-off contrast the post-run analysis uses. Ten sync pings are enough to place
 * 400 ms windows on the browser clock.
 */
async function locateLed(): Promise<void> {
  if (!rig.connected || !previewOn) return;
  locateBtn.disabled = true;
  const restoreWidthUs = readRigSettings().pulseWidthUs;
  try {
    roiEl.textContent = "blinking the LED…";
    await rig.setPulseWidth(LOCATE_WIDTH_US);
    const fit = fitSync(await rig.sync(LOCATE_SYNC_PINGS));
    const eventStart = rig.events.length;
    locateFrames = [];
    for (let i = 0; i < LOCATE_PULSES; i++) {
      await rig.pulse();
      await sleep(LOCATE_GAP_MS);
    }
    await sleep(300);
    const frames = locateFrames ?? [];
    locateFrames = null;
    const leds = toRigLeds(rig.events.slice(eventStart));
    panelRoi = chooseLedSignal(frames, leds, fit, PEAK_W, PEAK_H).roi;
    if (panelRoi) {
      roiEl.textContent =
        `LED contrast: ${num(panelRoi.signalContrast, 1)} counts ` +
        `(${num(panelRoi.signalMads, 1)} MADs) on ${panelRoi.plane}, ` +
        `${panelRoi.n} px at (${num(panelRoi.centroid.x, 1)}, ${num(panelRoi.centroid.y, 1)})`;
    } else {
      roiEl.textContent =
        `LED not found in ${frames.length} frames from ${leds.length} pulse(s) — ` +
        `check it is in frame and pointed at the lens; the difference view helps`;
    }
  } catch (err) {
    roiEl.textContent = `locate failed: ${errText(err)}`;
  } finally {
    locateFrames = null;
    try {
      await rig.setPulseWidth(restoreWidthUs);
    } catch {
      /* the rig may have gone away; the next run re-applies every setting anyway */
    }
    syncRigUi();
  }
}

function initRigPanel(): void {
  updateRigCounts();
  drawRigPlot(null);
  clearPreviewCanvas();
  // The preview is plain getUserMedia — no rig involved — but it exists to aim the rig's
  // LED, so its row only appears once one is connected.
  previewBtn.addEventListener("click", () => {
    if (previewOn) stopPreview();
    else void startPreview();
  });
  locateBtn.addEventListener("click", () => {
    void locateLed();
  });
  syncRigUi();
  if (!Rig.isSupported()) {
    setRigStatus("Web Serial unavailable — Chrome only; the rest of the page works without it");
    rigConnectBtn.disabled = true;
    return;
  }
  setRigStatus("not connected");
  rigConnectBtn.addEventListener("click", () => {
    void connectRig();
  });
  rigDisconnectBtn.addEventListener("click", () => {
    void disconnectRig();
  });
  rigSyncBtn.addEventListener("click", () => {
    void (async () => {
      rigSyncBtn.disabled = true;
      try {
        const fit = fitSync(await rig.sync(SYNC_PINGS));
        setRigStatus(
          `sync n ${fit.n}/${SYNC_PINGS}  RTT median ${num(fit.rttMedianMs, 2)} ms  ` +
            `residual sd ${num(fit.residualSdMs, 2)} ms  slope ${num((fit.b - 1) * 1e6, 0)} ppm`,
        );
      } catch (err) {
        setRigStatus(`sync failed: ${errText(err)}`);
      }
      rigSyncBtn.disabled = false;
    })();
  });
  rigPulseBtn.addEventListener("click", () => {
    void rig.pulse().catch((err: unknown) => setRigStatus(`pulse failed: ${errText(err)}`));
  });
}

// ---------------------------------------------------------------- test presets
//
// The three diagnostics, as buttons. Each writes the advanced controls and then starts an
// ordinary run, so advanced mode always shows exactly the settings that produced a result.

type TestId = "test1" | "test2" | "test3";

interface Preset {
  id: TestId;
  title: string;
  needsRig: boolean;
  button: HTMLButtonElement;
  card: HTMLElement;
  /** Writes the (hidden) advanced controls. */
  apply: () => void;
}

function setControl(id: string, value: string): void {
  el<HTMLInputElement>(id).value = value;
}

function setChecked(id: string, on: boolean): void {
  el<HTMLInputElement>(id).checked = on;
}

const PRESETS: Preset[] = [
  {
    id: "test1",
    title: "Test 1 — Loopback",
    needsRig: false,
    button: el<HTMLButtonElement>("preset-1"),
    card: el("sum-test1"),
    apply: () => {
      setControl("mode", "sparse");
      setControl("contrast", "full");
      setControl("duration", "15");
      setControl("gapMin", "500");
      setControl("gapMax", "1000");
      // Keep the LED quiet: this test is the loopback on its own. A connected rig still
      // timestamps the screen edges, which costs nothing and gives the display lag free.
      setControl("rig-mode", "0");
      setChecked("rig-train", false);
      setChecked("rig-edges", true);
    },
  },
  {
    id: "test2",
    title: "Test 2 — Rig relay validation",
    needsRig: true,
    button: el<HTMLButtonElement>("preset-2"),
    card: el("sum-test2"),
    apply: () => {
      setControl("mode", "sparse");
      setControl("contrast", "full");
      setControl("duration", "60");
      setControl("gapMin", "500");
      setControl("gapMax", "1000");
      setControl("rig-mode", "2");
      setControl("rig-delay-min", "100000");
      setControl("rig-delay-max", "140000");
      setControl("rig-pulse-width", "100000");
      setChecked("rig-train", false);
      setChecked("rig-edges", true);
    },
  },
  {
    id: "test3",
    title: "Test 3 — Exposure sweep",
    needsRig: true,
    button: el<HTMLButtonElement>("preset-3"),
    card: el("sum-test3"),
    apply: () => {
      setControl("mode", "sparse");
      setControl("contrast", "full");
      setControl("duration", "60");
      setControl("gapMin", "500");
      setControl("gapMax", "1000");
      setControl("rig-mode", "0");
      setControl("rig-pulse-width", "100000");
      setChecked("rig-train", true);
      setControl("rig-gap-min", "600");
      setControl("rig-gap-max", "1200");
      setChecked("rig-edges", true);
    },
  },
];

interface TestSummary {
  verdict: string;
  lines: string[];
  at: string;
}

/** Last result of each test, in memory only — nothing is persisted across reloads. */
const summaries = new Map<TestId, TestSummary>();

function verdictClass(v: string): string {
  if (v.startsWith("OK")) return "ok";
  if (v.startsWith("INCONCLUSIVE")) return "warnish";
  return "bad";
}

function renderSummaries(): void {
  for (const p of PRESETS) {
    const s = summaries.get(p.id);
    p.card.textContent = "";
    const title = document.createElement("div");
    title.className = "card-title";
    title.textContent = p.title;
    p.card.appendChild(title);
    if (!s) {
      const none = document.createElement("div");
      none.className = "card-verdict";
      none.textContent = "not run yet";
      p.card.appendChild(none);
      continue;
    }
    const verdict = document.createElement("div");
    verdict.className = verdictClass(s.verdict);
    verdict.textContent = s.verdict;
    p.card.appendChild(verdict);
    for (const line of s.lines) {
      const row = document.createElement("div");
      row.textContent = line;
      p.card.appendChild(row);
    }
    const at = document.createElement("div");
    at.className = "card-verdict";
    at.textContent = s.at;
    p.card.appendChild(at);
  }
}

function signed(v: number, d = 1): string {
  if (!Number.isFinite(v)) return "n/a";
  return (v >= 0 ? "+" : "") + v.toFixed(d);
}

function ledContrastLine(roi: RigRoiSummary): string {
  if (!roi.centroid)
    return `LED contrast: not localised — fell back to the brightest ${PEAK_N} redness pixels`;
  const counts = roi.plane === "red" ? roi.contrastRed : roi.contrastGray;
  const mads = roi.plane === "red" ? roi.madsRed : roi.madsGray;
  const plane = roi.plane === "red" ? "redness" : "gray";
  return `LED contrast ${num(counts, 0)} counts (${num(mads, 0)} MADs, ${plane})`;
}

/**
 * The exposure sweep has no residual to judge, so its verdict is about whether the sweep
 * itself resolved: enough pulses, and a window that is narrower than a frame and positive.
 */
function exposureVerdict(exp: ExposureResult, camIntervalMs: number): string {
  if (exp.ledNotDetected) return "UNRELIABLE — the LED never separated from the background";
  if (exp.n < 20) return `INCONCLUSIVE — only ${exp.n} pulses landed in an exposure; run longer`;
  if (!Number.isFinite(exp.windowMs))
    return `INCONCLUSIVE — only ${exp.nRamp} pulses landed on the ramp (need 6); run longer`;
  if (!(exp.r2 >= 0.5))
    return `UNRELIABLE — ramp fit r² ${num(exp.r2, 2)} over ${exp.nRamp} points; the fraction is not linear in phase`;
  const frame = Number.isFinite(camIntervalMs) && camIntervalMs > 0 ? camIntervalMs : 1000 / 30;
  if (!(exp.windowMs > 0)) return `UNRELIABLE — window width ${num(exp.windowMs, 1)} ms`;
  if (exp.windowMs > 1.5 * frame)
    return `UNRELIABLE — window ${num(exp.windowMs, 1)} ms is wider than a camera frame (${num(frame, 1)} ms)`;
  return "OK";
}

function summariseTest(
  testId: TestId,
  mode: Mode,
  results: Results,
  rigReport: RigReport | null,
): TestSummary {
  const at = `run at ${new Date().toLocaleTimeString()}`;
  if (testId === "test1") {
    const v = results.variants.find((x) => x.clock === "captureTime") ?? results.variants[0];
    if (!v) return { verdict: results.verdict, lines: ["no camera samples"], at };
    const halves = results.halves
      ? `halves ${num(estimateOf(results.halves.first, mode), 0)}/${num(estimateOf(results.halves.second, mode), 0)}`
      : "no half split";
    return {
      verdict: results.verdict,
      lines: [
        `Loopback lag ${num(estimateOf(v, mode), 1)} ms ` +
          `(plateau ±${num(v.plateauWidthMs / 2, 1)}, D=${num(v.peakCorr, 2)}, ${halves})`,
      ],
      at,
    };
  }
  if (!rigReport) return { verdict: "UNRELIABLE — no rig data for this run", lines: [], at };
  if (testId === "test2") {
    const relay = rigReport.relay;
    const p = relay.plateau;
    const all = allLagStats(rigReport.display);
    // Display lag is quantised by the refresh, so the modes are a refresh apart: cluster
    // with that gap and the two peaks the panel actually shows come out separately.
    const refresh = results.rafInterval.mean > 1 ? results.rafInterval.mean : 1000 / 60;
    const modes = clusterLags(
      rigReport.display.pairs.map((x) => x.lagMs),
      refresh,
    )
      .sort((a, b) => b.n - a.n)
      .slice(0, 3)
      .map((c) => `${num(c.centerMs, 0)} ms ×${c.n}`)
      .join(", ");
    const rows = rigReport.rowBands;
    const lines = [
      `LED plateau ${num(p.centerMs, 1)} ms vs loopback ${num(rigReport.loopbackLagMs, 1)} ms ` +
        `(Δ ${signed(rigReport.plateauDeltaMs)} ms, ${p.consistent}/${p.n} consistent)`,
      `Display lag mean ${num(all.meanMs, 0)} / min ${num(all.minMs, 0)} ms` +
        (modes ? ` — ${modes}` : ""),
      `Exposure ≈ ${num(relay.exposureEstimateMs, 0)} ms (from jump fractions)`,
    ];
    if (rows && rows.bands.length > 0)
      lines.push(
        `Rolling shutter readout ≈ ${num(rows.readoutMs, 0)} ms ` +
          `(bands ${rows.bands.map((b) => num(b.lagMs, 0)).join(" / ")} ms)`,
      );
    lines.push(`Mid-exposure ideal ≈ ${num(rigReport.midExposureIdealMs, 0)} ms`);
    lines.push(ledContrastLine(rigReport.roi));
    return { verdict: rigReport.verdict, lines, at };
  }
  const exp = rigReport.exposure;
  return {
    verdict: exposureVerdict(exp, results.cameraInterval.mean),
    lines: [
      `Exposure window ${num(exp.startMs, 0)} → ${num(exp.endMs, 0)} ms rel. captureTime ` +
        `(width ${num(exp.windowMs, 0)} ms; ${exp.n} pulses, ${exp.nRamp} on the ramp, r² ${num(exp.r2, 2)})`,
      `Camera frame period ${num(results.cameraInterval.mean, 1)} ms ` +
        `(${num(1000 / results.cameraInterval.mean, 1)} fps); LED on/off frames ` +
        `${rigReport.roi ? `${rigReport.roi.nOn}/${rigReport.roi.nOff}` : "n/a"} of ${rigReport.nLeds} pulses`,
      ledContrastLine(rigReport.roi),
    ],
    at,
  };
}

// ---------------------------------------------------------------- driver

/** One place for the camera request, so the aiming preview sees exactly what a run sees. */
const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
  audio: false,
};

let stream: MediaStream | null = null;

function releaseStream(): void {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
  video.srcObject = null;
  stream = null;
}

/** True while a run owns the camera and the screen; every other control waits. */
let runInProgress = false;

async function run(testId: TestId | null): Promise<void> {
  if (runInProgress) return;
  runInProgress = true;
  startBtn.disabled = true;
  syncRigUi();
  outEl.textContent = "…";
  rigOutEl.textContent = "no rig data.";
  jsonEl.value = "";

  const mode: Mode = el<HTMLSelectElement>("mode").value === "mseq" ? "mseq" : "sparse";
  const contrast = el<HTMLSelectElement>("contrast").value === "reduced" ? "reduced" : "full";
  // 20 % / 80 % sRGB gray keeps most of the step while dropping the full-white glare.
  const levels: [string, string] = contrast === "reduced" ? ["#333", "#ccc"] : ["#000", "#fff"];
  const durationS = Math.max(2, Number(el<HTMLInputElement>("duration").value) || 15);
  const symbolLen = Math.max(1, Math.round(Number(el<HTMLInputElement>("symbolLen").value) || 5));
  const order = Number(el<HTMLSelectElement>("order").value) === 8 ? 8 : 7;
  const gapMinMs = Math.max(200, Number(el<HTMLInputElement>("gapMin").value) || 500);
  const gapMaxMs = Math.max(gapMinMs, Number(el<HTMLInputElement>("gapMax").value) || 1000);
  const seed = (Math.random() * 2 ** 32) >>> 0;
  const schedule: Schedule =
    mode === "mseq"
      ? { kind: "mseq", seq: mSequence(order), symbolLen }
      : {
          kind: "sparse",
          times: sparseSchedule(durationS * 1000, gapMinMs, gapMaxMs, seededRandom(seed)),
        };

  // The run needs the video element and a stream of its own; the preview is holding both.
  stopPreview();
  syncRigUi();

  setStatus("requesting camera…");
  try {
    stream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);
  } catch (err) {
    setStatus("no camera");
    outEl.textContent = `camera unavailable: ${err instanceof Error ? err.message : String(err)}`;
    runInProgress = false;
    startBtn.disabled = false;
    syncRigUi();
    return;
  }
  const track = stream.getVideoTracks()[0];
  const camSettings = (track?.getSettings() ?? {}) as Record<string, unknown>;
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) {
    await new Promise<void>((resolve) =>
      video.addEventListener("loadedmetadata", () => resolve(), { once: true }),
    );
  }

  // Rig, if one is connected: configure it, then bracket the run with sync pings so the fit
  // interpolates over the whole run instead of extrapolating from one end of it.
  let rigRun: RigRun | null = null;
  if (rig.connected) {
    try {
      const rigSettings = readRigSettings();
      await applyRigSettings(rigSettings);
      const info = await rig.info().catch(() => null);
      setStatus("rig: syncing…");
      const eventStart = rig.events.length;
      const syncBefore = await rig.sync(SYNC_PINGS);
      rigRun = {
        settings: rigSettings,
        info,
        eventStart,
        syncBefore,
        syncAfter: [],
        events: [],
        planes: null,
      };
      setRigStatus(`armed — sync n ${syncBefore.length}/${SYNC_PINGS}`);
    } catch (err) {
      setRigStatus(`rig setup failed, running without it: ${errText(err)}`);
      rigRun = null;
    }
  }

  const panel = document.createElement("div");
  panel.id = "flash";
  panel.style.background = levels[0];
  document.body.appendChild(panel);

  try {
    for (let i = COUNTDOWN_S; i > 0; i--) {
      panel.textContent = String(i);
      setStatus(`starting in ${i}… (Esc aborts)`);
      await sleep(1000);
    }
    panel.textContent = "";
    setStatus("flashing… (Esc aborts)");

    const cap = await capture(panel, schedule, levels, durationS, rigRun !== null);

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    panel.remove();
    releaseStream();

    if (rigRun) {
      setStatus("rig: syncing…");
      try {
        await rig.setRelay(0);
        await rig.setTrain(false);
        rigRun.syncAfter = await rig.sync(SYNC_PINGS);
        await rig.setEdgeReporting(false);
        // The photodiode envelope is only meaningful once the screen has shown both levels,
        // so this is the reading worth showing the user.
        setConnectedStatus(await rig.info());
      } catch (err) {
        setRigStatus(`rig teardown failed: ${errText(err)}`);
      }
      rigRun.events = rig.events.slice(rigRun.eventStart);
      rigRun.planes = cap.planes;
    }
    setStatus("analysing…");

    const flips: Flip[] = cap.flips.map((f) => ({ t: f.t, level: f.level }));
    // Level changes, not flips: the first flip only pins the starting level.
    let nEdges = 0;
    for (let i = 1; i < flips.length; i++) if (flips[i].level !== flips[i - 1].level) nEdges++;
    const rafStats = intervalStats(cap.rafTimestamps);
    // A competing peak must sit outside the main lobe, whose half-width is one symbol.
    const symbolMs = mode === "mseq" && rafStats.n > 0 ? rafStats.mean * symbolLen : null;
    const lagOpts = symbolMs ? { secondPeakSeparationMs: 1.5 * symbolMs } : {};
    const captureSamples = pick(cap.samples, "captureTime");
    const camStats = intervalStats(captureSamples.map((s) => s.t));

    // captureTime is the clock the tracker actually stamps gaze samples with, so it drives the
    // verdict and the plotted curve; the other two are shown to expose how much the choice matters.
    const variants: VariantSummary[] = [];
    let primary: LagResult | null = null;
    for (const key of ["captureTime", "receiveTime", "now"] as const) {
      const lum = pick(cap.samples, key);
      if (lum.length < 10) continue;
      // Sparse holds outlast the webcam's auto-exposure loop, so each edge shows up as a jump
      // that decays back to baseline; the step-template correlation is biased low there. The
      // edge-difference estimator only looks at the jump itself.
      const r =
        mode === "sparse"
          ? estimateLagEdges(flips, lum, lagOpts)
          : estimateLag(flips, lum, lagOpts);
      variants.push(summarise(key, r));
      if (key === "captureTime" || primary === null) primary = r;
    }

    const halvesRaw =
      captureSamples.length >= 40
        ? splitHalves(
            flips,
            captureSamples,
            lagOpts,
            mode === "sparse" ? estimateLagEdges : estimateLag,
          )
        : null;

    const results: Results = {
      variants,
      halves: halvesRaw
        ? {
            first: summarise("first", halvesRaw.first),
            second: summarise("second", halvesRaw.second),
          }
        : null,
      cameraInterval: camStats,
      rafInterval: rafStats,
      droppedCameraFrames: droppedFrames(cap.samples),
      nFlips: cap.flips.length,
      verdict: verdictFor(mode, primary, halvesRaw, nEdges),
    };

    const settings: Settings = {
      mode,
      contrast,
      levels,
      durationS,
      gapMinMs: mode === "sparse" ? gapMinMs : null,
      gapMaxMs: mode === "sparse" ? gapMaxMs : null,
      seed: mode === "sparse" ? seed : null,
      symbolLen: mode === "mseq" ? symbolLen : null,
      order: mode === "mseq" ? order : null,
      sequenceLength: schedule.kind === "mseq" ? schedule.seq.length : null,
      symbolMsNominal: symbolMs,
      rvfc: cap.rvfc,
      videoWidth,
      videoHeight,
      exposureMode: typeof camSettings.exposureMode === "string" ? camSettings.exposureMode : null,
      frameRate: typeof camSettings.frameRate === "number" ? camSettings.frameRate : null,
    };

    outEl.textContent = renderReport(settings, results, cap);
    drawCurve(primary, mode);

    // The rig checks the number the loopback just produced, so it uses the same headline
    // estimate the report shows for the captureTime clock.
    const headline = variants.find((v) => v.clock === "captureTime") ?? variants[0];
    const rigReport =
      rigRun && headline
        ? analyseRig(rigRun, flips, cap.samples, estimateOf(headline, mode))
        : null;
    rigOutEl.textContent = rigReport ? renderRigReport(rigReport) : "no rig data.";
    drawRigPlot(rigReport);

    jsonEl.value = JSON.stringify(
      {
        userAgent: navigator.userAgent,
        screen: {
          width: window.screen.width,
          height: window.screen.height,
          devicePixelRatio: window.devicePixelRatio,
        },
        settings,
        results,
        flips: cap.flips,
        samples: cap.samples,
        rig: rigReport,
        note:
          "correlation curves omitted; they are recomputable from flips + samples. " +
          "The 160x120 planes the LED ROI was found in are not exported (~70 MB); " +
          "rig.roi.indices identifies the pixels used.",
      },
      null,
      0,
    );
    if (testId) {
      summaries.set(testId, summariseTest(testId, mode, results, rigReport));
      renderSummaries();
    }
    setStatus("done");
  } catch (err) {
    setStatus("error");
    outEl.textContent = `error: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`;
  } finally {
    // Idempotent: the happy path already tore these down before analysing.
    panel.remove();
    releaseStream();
    if (rigRun && rig.connected) {
      // An abort must not leave the LED relaying or the train running.
      void rig
        .setRelay(0)
        .then(() => rig.setTrain(false))
        .catch(() => undefined);
    }
    runInProgress = false;
    startBtn.disabled = false;
    startBtn.textContent = "Start with these settings";
    syncRigUi();
  }
}

function syncMode(): void {
  const fast = el<HTMLSelectElement>("mode").value === "mseq";
  // The flicker warning belongs to m-sequence mode, which only exists in advanced.
  el("warn-fast").hidden = !(fast && advanced);
  el("note-sparse").hidden = fast || !advanced;
  el("mseq-settings").hidden = !fast;
  el("sparse-settings").hidden = fast;
}
el<HTMLSelectElement>("mode").addEventListener("change", syncMode);

startBtn.addEventListener("click", () => {
  void run(null);
});

for (const preset of PRESETS) {
  preset.button.addEventListener("click", () => {
    // Write the advanced controls first, so what ran is visible there afterwards.
    preset.apply();
    syncMode();
    void run(preset.id);
  });
}

copyBtn.addEventListener("click", () => {
  if (!jsonEl.value) {
    setStatus("nothing to copy yet — run a test first");
    return;
  }
  // Selecting only works while the textarea is on screen (advanced mode).
  if (!jsonEl.hidden && jsonEl.offsetParent !== null) jsonEl.select();
  navigator.clipboard
    ?.writeText(jsonEl.value)
    .then(() => setStatus("JSON copied"))
    .catch(() => setStatus("copy failed — turn on Advanced, select the JSON and copy manually"));
});

drawCurve(null);
initAdvanced();
renderSummaries();
initRigPanel();
