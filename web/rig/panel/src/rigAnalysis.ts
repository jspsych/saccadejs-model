// Analysis for the hardware timing rig (see web/rig/README.md).
//
// The rig timestamps every 50 % luminance crossing on the screen with its own microsecond
// clock (`E` lines) and fires an LED into the webcam a programmable delay D after each edge
// (`L` lines), or on a free-running train. Three things come out of that, all measured
// during one ordinary sparse-mode loopback run:
//
//   1. `fitSync` maps rig microseconds onto `performance.now()` from the sync pings, so
//      rig events and browser events live on one timeline.
//   2. `displayLag` pairs each rAF flip with the photodiode edge it caused: the display's
//      own lag, separately for rising and falling edges (LCDs are much slower going dark).
//   3. `relayResiduals` closes the loop the loopback correction claims to close:
//      (captureTime of the frame the LED appears in − L) − flipTime − D should be zero.
//      Display lag and camera lag cancel by construction and no clock mapping enters the
//      left-hand side, so this is a direct test of the number applied to gaze data.
//   4. `exposureWindow` maps where the camera's integration window sits relative to
//      `captureTime`, from short LED pulses at uniformly random phase.
//
// Pure / DOM-free on purpose: the whole thing runs under vitest's node environment.

import { estimateLagEdges } from "./loopback";
import type { Flip, LumSample } from "./loopback";

export type { Flip };

// ---------------------------------------------------------------- types

export interface SyncPair {
  /** Token echoed by the rig, for reference. */
  token?: number;
  /** `performance.now()` immediately before the ping was written. */
  tSend: number;
  /** `performance.now()` when the reply line arrived. */
  tRecv: number;
  /** The rig's own clock at the moment it answered. */
  rigUs: number;
}

export interface SyncFit {
  /** browserMs = a + b * rigUs / 1000. */
  a: number;
  b: number;
  /** Pings kept after RTT outlier rejection. */
  n: number;
  rttMedianMs: number;
  rttMaxMs: number;
  residualSdMs: number;
}

export interface RigEdge {
  tUs: number;
  dir: "R" | "F";
  level: number;
  envMin: number;
  envMax: number;
}

export interface RigLed {
  tOnUs: number;
  tOffUs: number;
  src: "R" | "F" | "T" | "P";
  delayUs: number;
  edgeTUs: number;
}

/** One camera frame: the loopback's own stamp plus the LED detector's brightest-pixel mean. */
export interface CamFrame {
  captureTime: number;
  now: number;
  peak: number;
}

/**
 * The two per-pixel planes the LED can be looked for in. The rig's LED is red, so
 * `redness = clamp(R - max(G, B))` separates it from anything white — but a bright LED
 * blooms to white at its centre, where gray wins. Both are tried and the better one used.
 */
export type LedPlane = "gray" | "red";

/** A camera frame with its downscaled planes kept, for rig-informed LED localisation. */
export interface CamFrameGray extends CamFrame {
  gray: Uint8Array;
  red: Uint8Array;
  /** Brightest-N mean of the redness plane, the no-ROI fallback signal. */
  peakRed: number;
}

export interface RoiResult {
  plane: LedPlane;
  /** Pixels actually averaged (<= k). */
  n: number;
  /** The cap on the ROI size. */
  k: number;
  centroid: { x: number; y: number };
  /** mean(on) - mean(off) of the weakest pixel in the ROI, in plane counts. */
  contrast: number;
  /** Mean contrast over the ROI. */
  meanContrast: number;
  /** Contrast of the k-th ranked pixel, whether or not it made the ROI. */
  contrastKth: number;
  /** Separation of the resulting ROI signal, in counts and in baseline MADs. */
  signalContrast: number;
  signalMads: number;
  indices: number[];
  nOn: number;
  nOff: number;
  width: number;
  height: number;
}

export type LedSource = "roi-gray" | "roi-red" | "peak24-gray" | "peak24-red";

export interface LedSignalChoice {
  source: LedSource;
  /** The frames the rest of the analysis should use: `peak` is the chosen signal. */
  frames: CamFrame[];
  /** The winning ROI, or null when localisation failed and a brightest-N mean is in use. */
  roi: RoiResult | null;
  roiGray: RoiResult | null;
  roiRed: RoiResult | null;
}

export interface LagStats {
  n: number;
  meanMs: number;
  sdMs: number;
  minMs: number;
  maxMs: number;
}

export interface DisplayLagPair {
  flipT: number;
  edgeMs: number;
  lagMs: number;
  dir: "R" | "F";
}

export interface DisplayLagResult {
  rising: LagStats;
  falling: LagStats;
  pairs: DisplayLagPair[];
  unmatchedFlips: number;
  unmatchedEdges: number;
}

export interface DetectOptions {
  /**
   * Frames whose `captureTime` is within this of an LED window are kept out of the baseline:
   * a frame stamped just after a window still integrated part of it. Defaults to the median
   * inter-frame interval.
   */
  guardMs?: number;
  /** How far before the mapped LED on-time to start looking for the first lit frame. */
  searchBeforeMs?: number;
  /** How far after it to stop looking. */
  searchAfterMs?: number;
  /** Maximum flip → edge gap when attributing a relayed pulse to a flip. */
  maxFlipGapMs?: number;
}

export interface LedDetector {
  /** Median peak over frames far from every LED window. */
  baseline: number;
  /** Median peak over frames the LED lit completely. */
  ledLevel: number;
  /** Median absolute deviation of the baseline frames. */
  madBaseline: number;
  /** Halfway between baseline and ledLevel: "the LED is clearly in this frame". */
  threshold: number;
  /** Just above the noise floor: "this frame caught some of the LED". */
  noiseThreshold: number;
  nInside: number;
  nOutside: number;
  /** False when baseline and LED level are not separated by 3x the baseline MAD. */
  detected: boolean;
}

export interface RelayResidual {
  delayMs: number;
  residualMs: number;
  directLagMs: number;
  dir: "R" | "F";
  /** (peak - baseline) / (ledLevel - baseline) of the chosen frame, clamped: the partial-
   *  exposure fraction. Residual against f is the picture of where the half-frame goes. */
  f: number;
  /** The frame-to-frame rise that identified the frame, in signal counts. */
  jump: number;
  flipT: number;
  ledOnMs: number;
  captureTime: number;
  /** Stamp of the frame before it: with `captureTime`, the bracket on the LED's arrival. */
  prevCaptureTime: number;
}

/**
 * The set of lags consistent with every relayed pulse — the same intersection statistic the
 * loopback's edge estimator reports, so the two numbers can be compared directly.
 */
export interface RelayPlateau {
  centerMs: number;
  lowMs: number;
  highMs: number;
  widthMs: number;
  /** Pulses whose bracket contains the plateau. */
  consistent: number;
  n: number;
}

export interface RelayResult {
  n: number;
  residuals: RelayResidual[];
  /** The headline: apples-to-apples with the loopback estimate. */
  plateau: RelayPlateau;
  /**
   * Mean of `directLagMs`. NOT a bias test: the first lit frame is stamped up to a frame
   * after the event, so this mean sits about half a frame period above the true lag by
   * construction. Kept because its spread is a useful sanity check.
   */
  meanResidualMs: number;
  sdResidualMs: number;
  meanDirectLagMs: number;
  sdDirectLagMs: number;
  /** Exposure time inferred from how often the first lit frame caught the whole pulse. */
  exposureEstimateMs: number;
  /** 90th percentile of the per-pulse jumps: one frame fully lit. */
  fullJump: number;
  /** Fraction of pulses whose first lit frame jumped by >= 0.9 of that. */
  fullJumpFraction: number;
  /** Median camera frame interval used for the exposure estimate. */
  framePeriodMs: number;
  /** Relayed pulses that could not be tied to both a flip and a lit camera frame. */
  unmatched: number;
  ledNotDetected: boolean;
  detector: LedDetector;
}

export interface RowBand {
  /** Centre row of the band in the 160x120 frame. */
  row: number;
  lagMs: number;
  widthMs: number;
  /** D at the plateau: 1.0 = every edge's full jump captured in this band. */
  peak: number;
  n: number;
}

export interface RowBandResult {
  bands: RowBand[];
  /** Fit of band lag against band centre row. Negative = lower rows are read later. */
  slopeMsPerRow: number;
  /** |slope| x height: how much later the last row is captured than the first. */
  readoutMs: number;
  direction: "top-first" | "bottom-first" | "flat";
  ledRow: number | null;
  /** The fit evaluated at the LED's row, and at the middle row. */
  ledRowLagMs: number;
  midRowLagMs: number;
  wholeFrameLagMs: number;
  /** wholeFrameLagMs - ledRowLagMs: the row offset between screen- and LED-based lags. */
  offsetMs: number;
}

export interface ExposurePoint {
  phaseMs: number;
  f: number;
}

export interface ExposureResult {
  n: number;
  points: ExposurePoint[];
  /**
   * Earlier edge of the integration window, as a phase (LED onset − captureTime): where the
   * fitted ramp reaches f = 1. Frames integrate (captureTime + startMs, captureTime + endMs).
   */
  startMs: number;
  /** Later edge: where the fitted ramp reaches f = 0. */
  endMs: number;
  windowMs: number;
  /** Points on the ramp (0.15 ≤ f ≤ 0.85) the line was fitted through. */
  nRamp: number;
  /** Coefficient of determination of that fit; NaN when there was no fit. */
  r2: number;
  ledNotDetected: boolean;
  detector: LedDetector;
}

/** Ramp points closer than this to 0 or 1 are dropped from the line fit: the first-lit-frame
 *  detector cannot see a fraction below its noise floor, so the bottom of the ramp is
 *  truncated by construction, and the top is flat (fully lit) rather than linear. */
const RAMP_LO = 0.15;
const RAMP_HI = 0.85;
const MIN_RAMP_POINTS = 6;

const DEFAULT_MAX_LAG_MS = 250;
const DEFAULT_SEARCH_BEFORE_MS = 60;
const DEFAULT_SEARCH_AFTER_MS = 400;
const DEFAULT_GUARD_MS = 40;
/** Below this many completely lit frames, the LED level falls back to a high quantile. */
const MIN_INSIDE_FRAMES = 3;
/** ROI size cap: a 5 mm LED and its bloom at 160x120. */
const DEFAULT_ROI_K = 16;
/** ROI pixels must reach this fraction of the best pixel's contrast. */
const ROI_MIN_FRACTION = 0.25;
/** The best pixel must stand this many MADs above the middle of the contrast map. */
const ROI_MIN_CONTRAST_MADS = 8;
const PLATEAU_SCAN_MAX_MS = 600;
const PLATEAU_SCAN_STEP_MS = 0.5;
/** A pulse whose first lit frame jumped this much of a full one caught the whole onset. */
const FULL_JUMP_FRACTION = 0.9;
const DEFAULT_ROW_BANDS = 4;
/** A frame counts as "caught some LED" this far above the baseline, at minimum. */
const NOISE_FLOOR_FRACTION = 0.02;

// ---------------------------------------------------------------- small helpers

function median(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = xs.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function quantile(xs: number[], q: number): number {
  if (xs.length === 0) return NaN;
  const s = xs.slice().sort((a, b) => a - b);
  const i = (s.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
}

function stats(xs: number[]): LagStats {
  const n = xs.length;
  if (n === 0) return { n: 0, meanMs: NaN, sdMs: NaN, minMs: NaN, maxMs: NaN };
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of xs) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / n;
  let ss = 0;
  for (const v of xs) ss += (v - mean) * (v - mean);
  return { n, meanMs: mean, sdMs: n > 1 ? Math.sqrt(ss / (n - 1)) : 0, minMs: min, maxMs: max };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// ---------------------------------------------------------------- sync fit

/**
 * Least-squares map from rig microseconds to the browser clock.
 *
 * Each ping brackets one rig timestamp between the write and the reply, so the midpoint
 * `(tSend + tRecv) / 2` is the browser-side time of `rigUs` up to the link's asymmetry —
 * ±0.5 ms for a ~1 ms USB round trip. Pings whose RTT exceeds twice the median are dropped
 * (a scheduler hiccup on either side moves the true instant away from the midpoint).
 *
 * The fit is centred on the mean rig time before solving, so the huge absolute microsecond
 * values do not eat the available precision; `a` is still reported at rigUs = 0, which is a
 * long extrapolation — judge the fit by `residualSdMs`, and map times with `toBrowserMs`.
 */
export function fitSync(pairs: SyncPair[]): SyncFit {
  const rtts = pairs.map((p) => p.tRecv - p.tSend);
  const rttMedianMs = median(rtts);
  const rttMaxMs = rtts.length > 0 ? Math.max(...rtts) : NaN;
  const kept =
    Number.isFinite(rttMedianMs) && rttMedianMs > 0
      ? pairs.filter((p) => p.tRecv - p.tSend <= 2 * rttMedianMs)
      : pairs.slice();

  const n = kept.length;
  if (n === 0) return { a: NaN, b: NaN, n: 0, rttMedianMs, rttMaxMs, residualSdMs: NaN };

  const xs = kept.map((p) => p.rigUs / 1000);
  const ys = kept.map((p) => (p.tSend + p.tRecv) / 2);
  if (n === 1) {
    return { a: ys[0] - xs[0], b: 1, n, rttMedianMs, rttMaxMs, residualSdMs: 0 };
  }

  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 1;
  const a = my - b * mx;

  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += r * r;
  }
  const residualSdMs = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { a, b, n, rttMedianMs, rttMaxMs, residualSdMs };
}

/** Rig microseconds → `performance.now()` milliseconds under `fit`. */
export function toBrowserMs(fit: SyncFit, rigUs: number): number {
  return fit.a + (fit.b * rigUs) / 1000;
}

// ---------------------------------------------------------------- display lag

/** Direction of the photon change a flip causes: to level 1 is a rising edge. */
function flipDir(level: number): "R" | "F" {
  return level === 1 ? "R" : "F";
}

/**
 * Display lag per flip: the delay from the rAF timestamp that painted a level to the
 * photodiode's 50 % crossing.
 *
 * The first flip is skipped — it only pins the initial level and the panel was already
 * showing it, so no photons changed. Each rig edge is consumed at most once, so a missed
 * edge cannot make every later flip match the wrong one.
 */
export function displayLag(
  flips: Flip[],
  edges: RigEdge[],
  fit: SyncFit,
  maxLagMs = DEFAULT_MAX_LAG_MS,
): DisplayLagResult {
  const sortedFlips = flips.slice().sort((a, b) => a.t - b.t);
  const mapped = edges
    .map((e) => ({ ms: toBrowserMs(fit, e.tUs), dir: e.dir }))
    .filter((e) => Number.isFinite(e.ms))
    .sort((a, b) => a.ms - b.ms);
  const used = new Array<boolean>(mapped.length).fill(false);

  const pairs: DisplayLagPair[] = [];
  let unmatchedFlips = 0;
  for (let i = 1; i < sortedFlips.length; i++) {
    const f = sortedFlips[i];
    if (f.level === sortedFlips[i - 1].level) continue;
    const dir = flipDir(f.level);
    let hit = -1;
    for (let k = 0; k < mapped.length; k++) {
      const m = mapped[k];
      if (m.ms <= f.t) continue;
      if (m.ms - f.t > maxLagMs) break;
      if (used[k] || m.dir !== dir) continue;
      hit = k;
      break;
    }
    if (hit < 0) {
      unmatchedFlips++;
      continue;
    }
    used[hit] = true;
    pairs.push({ flipT: f.t, edgeMs: mapped[hit].ms, lagMs: mapped[hit].ms - f.t, dir });
  }

  const unmatchedEdges = used.filter((u) => !u).length;
  return {
    rising: stats(pairs.filter((p) => p.dir === "R").map((p) => p.lagMs)),
    falling: stats(pairs.filter((p) => p.dir === "F").map((p) => p.lagMs)),
    pairs,
    unmatchedFlips,
    unmatchedEdges,
  };
}

// ---------------------------------------------------------------- LED detection

interface LedWindow {
  onMs: number;
  offMs: number;
}

function ledWindows(leds: RigLed[], fit: SyncFit): LedWindow[] {
  return leds
    .map((l) => ({ onMs: toBrowserMs(fit, l.tOnUs), offMs: toBrowserMs(fit, l.tOffUs) }))
    .filter((w) => Number.isFinite(w.onMs) && Number.isFinite(w.offMs) && w.offMs >= w.onMs)
    .sort((a, b) => a.onMs - b.onMs);
}

/**
 * The classification of a frame against the LED windows is asymmetric, because
 * `captureTime` marks the *end* of an exposure, not the middle of it.
 *
 * A frame stamped at t integrated roughly (t − E, t], where E is the exposure — at most one
 * frame interval, which is what we have to work with. So:
 *
 *   fully lit   onMs + E <= t <= offMs      (the whole exposure fell inside the pulse)
 *   fully dark  t <= onMs  or  t >= offMs + E, for every window
 *   partial     anything else — excluded from both sets
 *
 * Guarding both ends of the window, as an earlier version did, throws away most of a short
 * pulse: at 23 fps a 100 ms pulse minus 43 ms at each end leaves a 14 ms slot, so most
 * pulses contribute no lit frame at all and the LED "disappears".
 */
function litWindow(windows: LedWindow[], t: number, guardMs: number): LedWindow | null {
  for (const w of windows) {
    if (t < w.onMs + guardMs) break;
    if (t <= w.offMs) return w;
  }
  return null;
}

/** True when no LED window could have put any light into the frame stamped at `t`. */
function isDark(windows: LedWindow[], t: number, guardMs: number): boolean {
  for (const w of windows) {
    if (t <= w.onMs) break;
    if (t < w.offMs + guardMs) return false;
  }
  return true;
}

function frameInterval(frames: CamFrame[]): number {
  const d: number[] = [];
  for (let i = 1; i < frames.length; i++) d.push(frames[i].captureTime - frames[i - 1].captureTime);
  const m = median(d);
  return Number.isFinite(m) && m > 0 ? m : DEFAULT_GUARD_MS;
}

/**
 * Baseline and lit levels of the LED detector's per-frame `peak`, over all the data.
 *
 * `baseline` is the median peak of frames that are fully dark (`isDark`); `ledLevel` the
 * median of frames the LED lit for their whole integration (`litWindow`: stamped at least one
 * frame interval after the pulse began and no later than its end). Pulses shorter than a
 * frame never produce such a frame, so below MIN_INSIDE_FRAMES of them the level falls back
 * to a high quantile of the frames that caught *any* LED — for a pulse at uniformly random
 * phase most land wholly inside one exposure, so the upper decile is the full-brightness
 * level.
 */
export function ledDetector(
  leds: RigLed[],
  frames: CamFrame[],
  fit: SyncFit,
  opts: DetectOptions = {},
): LedDetector {
  const sorted = frames.slice().sort((a, b) => a.captureTime - b.captureTime);
  const guardMs = opts.guardMs ?? frameInterval(sorted);
  const windows = ledWindows(leds, fit);

  const outside: number[] = [];
  const inside: number[] = [];
  const near: number[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const t = sorted[i].captureTime;
    if (isDark(windows, t, guardMs)) {
      outside.push(sorted[i].peak);
      continue;
    }
    near.push(sorted[i].peak);
    if (litWindow(windows, t, guardMs)) inside.push(sorted[i].peak);
  }

  const baseline = median(outside);
  const dev = outside.map((p) => Math.abs(p - baseline));
  const madBaseline = outside.length > 0 ? median(dev) : NaN;

  let ledLevel: number;
  if (inside.length >= MIN_INSIDE_FRAMES) {
    ledLevel = median(inside);
  } else {
    const floor = Number.isFinite(madBaseline) ? 3 * madBaseline : 0;
    const lit = near.filter((p) => p > baseline + Math.max(floor, 1e-9));
    ledLevel = lit.length >= MIN_INSIDE_FRAMES ? quantile(lit, 0.9) : quantile(near, 1);
  }

  const separation = ledLevel - baseline;
  const detected =
    Number.isFinite(baseline) &&
    Number.isFinite(ledLevel) &&
    separation > 0 &&
    separation >= 3 * (Number.isFinite(madBaseline) ? madBaseline : 0);
  const threshold = baseline + 0.5 * separation;
  const noiseThreshold =
    baseline +
    Math.max(
      3 * (Number.isFinite(madBaseline) ? madBaseline : 0),
      NOISE_FLOOR_FRACTION * separation,
    );
  return {
    baseline,
    ledLevel,
    madBaseline,
    threshold,
    noiseThreshold,
    nInside: inside.length,
    nOutside: outside.length,
    detected,
  };
}

/**
 * Where the LED actually is in the frame, from the rig's own record of when it was lit.
 *
 * The brightest-N-pixels detector is a guess that a face highlight or a desk lamp can win:
 * a small LED can sit only a few counts above them, which is what makes the first-lit frame
 * ambiguous. But the rig knows exactly when the LED was on, so the LED is simply the set of
 * pixels that differ between the on-frames and the off-frames. Both sets are taken with a
 * guard of one frame interval at each end, so a frame that straddles an edge of a window
 * lands in neither.
 *
 * The ROI is the top `k` pixels by that contrast, minus any that fall below a quarter of the
 * best pixel's contrast: a 5 mm LED plus its bloom covers a dozen-odd pixels at 160x120, but
 * padding the ROI out to a fixed `k` with near-zero-contrast pixels would both dilute the
 * signal and put the reported centroid somewhere between the LED and a random pixel.
 * `contrastKth` still reports the k-th ranked contrast, as a check on the ROI's size.
 */
export function localiseLed(
  frames: CamFrameGray[],
  leds: RigLed[],
  fit: SyncFit,
  width: number,
  height: number,
  opts: DetectOptions & { k?: number; plane?: LedPlane } = {},
): RoiResult | null {
  const npix = width * height;
  if (npix <= 0) return null;
  const plane: LedPlane = opts.plane ?? "gray";
  const planeOf = (f: CamFrameGray): Uint8Array => (plane === "red" ? f.red : f.gray);
  const sorted = frames.slice().sort((a, b) => a.captureTime - b.captureTime);
  const guardMs = opts.guardMs ?? frameInterval(sorted);
  const windows = ledWindows(leds, fit);
  if (windows.length === 0) return null;

  const sumOn = new Float64Array(npix);
  const sumOff = new Float64Array(npix);
  let nOn = 0;
  let nOff = 0;
  for (const f of sorted) {
    const p = planeOf(f);
    if (p.length < npix) continue;
    if (litWindow(windows, f.captureTime, guardMs)) {
      for (let i = 0; i < npix; i++) sumOn[i] += p[i];
      nOn++;
    } else if (isDark(windows, f.captureTime, guardMs)) {
      for (let i = 0; i < npix; i++) sumOff[i] += p[i];
      nOff++;
    }
  }
  if (nOn < MIN_INSIDE_FRAMES || nOff < MIN_INSIDE_FRAMES) return null;

  const contrast = new Float64Array(npix);
  for (let i = 0; i < npix; i++) contrast[i] = sumOn[i] / nOn - sumOff[i] / nOff;

  const order = new Uint32Array(npix);
  for (let i = 0; i < npix; i++) order[i] = i;
  order.sort((a, b) => contrast[b] - contrast[a]);

  const k = Math.max(1, Math.min(opts.k ?? DEFAULT_ROI_K, npix));
  const best = contrast[order[0]];
  if (!(best > 0)) return null;
  // The brightest pixel of a *noise* contrast map still stands ~6 MADs above the middle of
  // it, purely because there are 19200 of them. A real LED stands hundreds. Reject anything
  // that only looks like the tail of the noise, so "no LED in frame" reports as such rather
  // than as a confident ROI on a random pixel.
  const mid = contrast[order[npix >> 1]];
  const madContrast = median(Array.from(contrast, (c) => Math.abs(c - mid)));
  if (madContrast > 0 && best < ROI_MIN_CONTRAST_MADS * madContrast) return null;
  const floor = best * ROI_MIN_FRACTION;
  const indices: number[] = [];
  for (let i = 0; i < k; i++) {
    if (contrast[order[i]] < floor) break;
    indices.push(order[i]);
  }

  let cx = 0;
  let cy = 0;
  let sum = 0;
  for (const i of indices) {
    cx += i % width;
    cy += Math.floor(i / width);
    sum += contrast[i];
  }
  const roi: RoiResult = {
    plane,
    n: indices.length,
    k,
    centroid: { x: cx / indices.length, y: cy / indices.length },
    contrast: contrast[indices[indices.length - 1]],
    meanContrast: sum / indices.length,
    contrastKth: contrast[order[k - 1]],
    signalContrast: NaN,
    signalMads: NaN,
    indices,
    nOn,
    nOff,
    width,
    height,
  };
  // How well the ROI signal itself separates, which is what the two planes compete on.
  const det = ledDetector(leds, roiFrames(sorted, roi), fit, opts);
  roi.signalContrast = det.ledLevel - det.baseline;
  roi.signalMads =
    det.madBaseline > 0
      ? roi.signalContrast / det.madBaseline
      : roi.signalContrast > 0
        ? Infinity
        : 0;
  return roi;
}

/** Mean of the ROI pixels in one frame of the ROI's plane — the LED signal the analysis uses. */
export function roiMean(plane: Uint8Array, roi: RoiResult): number {
  let sum = 0;
  for (const i of roi.indices) sum += plane[i];
  return sum / roi.indices.length;
}

/** The same frames with `peak` replaced by the ROI mean on the ROI's plane. */
export function roiFrames(frames: CamFrameGray[], roi: RoiResult): CamFrame[] {
  return frames.map((f) => ({
    captureTime: f.captureTime,
    now: f.now,
    peak: roiMean(roi.plane === "red" ? f.red : f.gray, roi),
  }));
}

/**
 * Pick the LED signal the rest of the analysis runs on.
 *
 * Rig-informed ROIs on both planes first, judged by how far the resulting signal's lit level
 * sits above its baseline in units of that baseline's MAD — the quantity that decides whether
 * a lit frame can be told from an unlit one at all. If neither plane localises (too few
 * pulses, or no LED in frame) it falls back to the brightest-N mean of the redness plane,
 * which still rejects a white desk lamp, unlike the gray one.
 */
export function chooseLedSignal(
  frames: CamFrameGray[],
  leds: RigLed[],
  fit: SyncFit,
  width: number,
  height: number,
  opts: DetectOptions & { k?: number } = {},
): LedSignalChoice {
  const roiGray = localiseLed(frames, leds, fit, width, height, { ...opts, plane: "gray" });
  const roiRed = localiseLed(frames, leds, fit, width, height, { ...opts, plane: "red" });
  const score = (r: RoiResult | null): number =>
    r && Number.isFinite(r.signalMads) ? r.signalMads : r ? 1e9 : -Infinity;
  const roi = score(roiRed) > score(roiGray) ? roiRed : roiGray;
  if (roi) {
    return {
      source: roi.plane === "red" ? "roi-red" : "roi-gray",
      frames: roiFrames(frames, roi),
      roi,
      roiGray,
      roiRed,
    };
  }
  return {
    source: "peak24-red",
    frames: frames.map((f) => ({ captureTime: f.captureTime, now: f.now, peak: f.peakRed })),
    roi: null,
    roiGray,
    roiRed,
  };
}

/** First frame in [tOnMs - before, tOnMs + after] whose peak clears `threshold`; -1 if none. */
function firstLitFrame(
  frames: CamFrame[],
  tOnMs: number,
  threshold: number,
  beforeMs: number,
  afterMs: number,
): number {
  for (let i = 0; i < frames.length; i++) {
    const t = frames[i].captureTime;
    if (t < tOnMs - beforeMs) continue;
    if (t > tOnMs + afterMs) break;
    if (frames[i].peak > threshold) return i;
  }
  return -1;
}

/**
 * The frame that caught the LED coming on: the biggest frame-to-frame rise of the signal in
 * the search window — the same convention the loopback's edge estimator uses, and unlike a
 * fixed threshold it does not care what fraction of the exposure the LED was on for. The
 * threshold survives only as a sanity gate: the rise must clear the noise floor.
 */
function bestJumpFrame(
  frames: CamFrame[],
  tOnMs: number,
  beforeMs: number,
  afterMs: number,
  minJump: number,
): { i: number; jump: number } {
  let best = -1;
  let bestJump = -Infinity;
  for (let i = 1; i < frames.length; i++) {
    const t = frames[i].captureTime;
    if (t < tOnMs - beforeMs) continue;
    if (t > tOnMs + afterMs) break;
    const jump = frames[i].peak - frames[i - 1].peak;
    if (jump > bestJump) {
      bestJump = jump;
      best = i;
    }
  }
  return best >= 0 && bestJump > minJump ? { i: best, jump: bestJump } : { i: -1, jump: NaN };
}

/**
 * The lags consistent with every relayed pulse.
 *
 * A pulse observed in frame j says only this: the LED's light arrived after frame j-1 was
 * stamped and no later than frame j's stamp. So the lags τ compatible with that one pulse
 * are the half-open interval (c_{j-1} − t_flip − D, c_j − t_flip − D]. Intersecting those
 * intervals over all pulses is exactly what the loopback's edge estimator does with screen
 * edges, which is the point: the two numbers are then measuring the same thing and can be
 * subtracted. The mean of the per-pulse `directLagMs` cannot be compared that way — it sits
 * half a frame high by construction, because a stamp always lands after the event.
 */
export function relayPlateau(
  rows: RelayResidual[],
  opts: { minMs?: number; maxMs?: number; stepMs?: number } = {},
): RelayPlateau {
  const min = opts.minMs ?? 0;
  const max = opts.maxMs ?? PLATEAU_SCAN_MAX_MS;
  const step = opts.stepMs ?? PLATEAU_SCAN_STEP_MS;
  const empty = {
    centerMs: NaN,
    lowMs: NaN,
    highMs: NaN,
    widthMs: NaN,
    consistent: 0,
    n: rows.length,
  };
  if (rows.length === 0 || !(step > 0) || max < min) return empty;

  const lo = rows.map((r) => r.prevCaptureTime - r.flipT - r.delayMs);
  const hi = rows.map((r) => r.captureTime - r.flipT - r.delayMs);
  const steps = Math.round((max - min) / step);
  const counts = new Int32Array(steps + 1);
  let best = 0;
  for (let i = 0; i <= steps; i++) {
    const tau = min + i * step;
    let c = 0;
    for (let k = 0; k < rows.length; k++) if (lo[k] < tau && tau <= hi[k]) c++;
    counts[i] = c;
    if (c > best) best = c;
  }
  if (best === 0) return empty;

  // Longest contiguous run at the maximum, so a lone tied grid point cannot define it.
  let runStart = -1;
  let bestStart = -1;
  let bestLen = 0;
  for (let i = 0; i <= steps + 1; i++) {
    const at = i <= steps && counts[i] === best;
    if (at && runStart < 0) runStart = i;
    if (!at && runStart >= 0) {
      const len = i - runStart;
      if (len > bestLen) {
        bestLen = len;
        bestStart = runStart;
      }
      runStart = -1;
    }
  }
  const lowMs = min + bestStart * step;
  const highMs = min + (bestStart + bestLen - 1) * step;
  return {
    centerMs: (lowMs + highMs) / 2,
    lowMs,
    highMs,
    widthMs: highMs - lowMs,
    consistent: best,
    n: rows.length,
  };
}

// ---------------------------------------------------------------- relay residuals

/**
 * The closed-loop test of the loopback correction.
 *
 * For every LED pulse relayed from a screen edge: the rig reports the delay D it applied,
 * the browser knows the rAF flip that caused the edge and the `captureTime` c of the first
 * frame the LED is visible in. Display lag and camera lag are both inside c, and both are
 * inside the loopback estimate L, so
 *
 *     residual = (c − L) − t_flip − D
 *
 * is zero when the correction is right, with a spread set by the frame quantisation and the
 * exposure window. `directLagMs = c − t_flip − D` is the same measurement without L: an
 * estimate of the loopback lag that never touches the loopback estimator (nor, on that side,
 * the rig→browser clock fit — only D and the flip time enter it).
 */
export function relayResiduals(
  leds: RigLed[],
  edges: RigEdge[],
  flips: Flip[],
  frames: CamFrame[],
  fit: SyncFit,
  loopbackLagMs: number,
  opts: DetectOptions = {},
): RelayResult {
  const detector = ledDetector(leds, frames, fit, opts);
  const sortedFrames = frames.slice().sort((a, b) => a.captureTime - b.captureTime);
  const sortedFlips = flips.slice().sort((a, b) => a.t - b.t);
  const before = opts.searchBeforeMs ?? DEFAULT_SEARCH_BEFORE_MS;
  const after = opts.searchAfterMs ?? DEFAULT_SEARCH_AFTER_MS;
  const maxGap = opts.maxFlipGapMs ?? DEFAULT_MAX_LAG_MS;
  const relayed = leds.filter((l) => l.src === "R" || l.src === "F");
  const separation = detector.ledLevel - detector.baseline;
  const minJump = detector.noiseThreshold - detector.baseline;

  const residuals: RelayResidual[] = [];
  let unmatched = 0;

  if (detector.detected) {
    for (const led of relayed) {
      const dir = led.src === "R" ? "R" : "F";
      const edge = edges.find((e) => e.tUs === led.edgeTUs);
      const edgeMs = toBrowserMs(fit, led.edgeTUs);
      const wantDir = edge ? edge.dir : dir;

      // The flip that produced this edge: the nearest one before it, same direction.
      let flip: Flip | null = null;
      for (let i = sortedFlips.length - 1; i >= 1; i--) {
        const f = sortedFlips[i];
        if (f.t > edgeMs) continue;
        if (edgeMs - f.t > maxGap) break;
        if (flipDir(f.level) !== wantDir) continue;
        flip = f;
        break;
      }
      const tOnMs = toBrowserMs(fit, led.tOnUs);
      const hit =
        flip === null
          ? { i: -1, jump: NaN }
          : bestJumpFrame(sortedFrames, tOnMs, before, after, minJump);
      if (flip === null || hit.i < 0) {
        unmatched++;
        continue;
      }
      const frame = sortedFrames[hit.i];
      const c = frame.captureTime;
      const delayMs = led.delayUs / 1000;
      const directLagMs = c - flip.t - delayMs;
      residuals.push({
        delayMs,
        residualMs: directLagMs - loopbackLagMs,
        directLagMs,
        dir,
        // How much of that frame's exposure the LED was on for, roughly: with a long
        // exposure a residual near zero should come with f near 1.
        f: separation > 0 ? clamp01((frame.peak - detector.baseline) / separation) : NaN,
        jump: hit.jump,
        flipT: flip.t,
        ledOnMs: tOnMs,
        captureTime: c,
        prevCaptureTime: sortedFrames[hit.i - 1].captureTime,
      });
    }
  } else {
    unmatched = relayed.length;
  }

  const res = stats(residuals.map((r) => r.residualMs));
  const direct = stats(residuals.map((r) => r.directLagMs));
  // Exposure from the jump sizes: a pulse that starts during a frame's integration splits
  // its onset over two frames and the first one jumps short. That happens for a fraction
  // E/P of pulses, so the fraction that jump the full amount is (P - E)/P.
  const jumps = residuals.map((r) => r.jump).filter((j) => Number.isFinite(j));
  const fullJump = jumps.length > 0 ? quantile(jumps, 0.9) : NaN;
  const fullCount = jumps.filter((j) => j >= FULL_JUMP_FRACTION * fullJump).length;
  const fullJumpFraction = jumps.length > 0 ? fullCount / jumps.length : NaN;
  const framePeriodMs = frameInterval(sortedFrames);
  return {
    plateau: relayPlateau(residuals),
    exposureEstimateMs: (1 - fullJumpFraction) * framePeriodMs,
    fullJump,
    fullJumpFraction,
    framePeriodMs,
    n: residuals.length,
    residuals,
    meanResidualMs: res.meanMs,
    sdResidualMs: res.sdMs,
    meanDirectLagMs: direct.meanMs,
    sdDirectLagMs: direct.sdMs,
    unmatched,
    ledNotDetected: !detector.detected,
    detector,
  };
}

// ---------------------------------------------------------------- rolling shutter

/**
 * Does the camera read the frame a row at a time?
 *
 * A rolling shutter gives every row its own instant while the whole frame carries one
 * `captureTime`, so the lag measured from the screen flashes depends on where in the frame
 * you measure it. Splitting the stored gray plane into horizontal bands and running the
 * loopback's own edge estimator on each turns that into a number: the slope of band lag
 * against band row, times the frame height, is how much later the last row is captured than
 * the first. It also says how much of the gap between the screen-based loopback lag (whole
 * frame) and the LED-based one (a few rows, wherever the LED sits) is just this.
 */
export function rowBands(
  frames: CamFrameGray[],
  flips: Flip[],
  width: number,
  height: number,
  wholeFrameLagMs: number,
  ledRow: number | null,
  nBands = DEFAULT_ROW_BANDS,
): RowBandResult {
  const empty: RowBandResult = {
    bands: [],
    slopeMsPerRow: NaN,
    readoutMs: NaN,
    direction: "flat",
    ledRow,
    ledRowLagMs: NaN,
    midRowLagMs: NaN,
    wholeFrameLagMs,
    offsetMs: NaN,
  };
  const npix = width * height;
  const usable = frames.filter((f) => f.gray.length >= npix);
  if (usable.length < 10 || flips.length < 2 || nBands < 2) return empty;

  const rowsPer = Math.floor(height / nBands);
  if (rowsPer < 1) return empty;
  const series: LumSample[][] = Array.from({ length: nBands }, () => []);
  for (const f of usable) {
    for (let b = 0; b < nBands; b++) {
      const start = b * rowsPer * width;
      const end = start + rowsPer * width;
      let sum = 0;
      for (let i = start; i < end; i++) sum += f.gray[i];
      series[b].push({ t: f.captureTime, lum: sum / (rowsPer * width) });
    }
  }

  const bands: RowBand[] = series.map((samples, b) => {
    const r = estimateLagEdges(flips, samples);
    return {
      row: b * rowsPer + (rowsPer - 1) / 2,
      lagMs: r.plateau.centerMs,
      widthMs: r.plateau.widthMs,
      peak: r.peakCorr,
      n: r.nSamples,
    };
  });

  const good = bands.filter((b) => Number.isFinite(b.lagMs));
  if (good.length < 2) return { ...empty, bands };
  let mx = 0;
  let my = 0;
  for (const b of good) {
    mx += b.row;
    my += b.lagMs;
  }
  mx /= good.length;
  my /= good.length;
  let sxx = 0;
  let sxy = 0;
  for (const b of good) {
    const dx = b.row - mx;
    sxx += dx * dx;
    sxy += dx * (b.lagMs - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  const at = (row: number): number => intercept + slope * row;
  const ledRowLagMs = ledRow === null ? NaN : at(ledRow);
  return {
    bands,
    slopeMsPerRow: slope,
    readoutMs: Math.abs(slope) * height,
    // Lower rows read later means the edge reaches them in an earlier frame, so their
    // fitted lag is smaller: a negative slope is the ordinary top-to-bottom readout.
    direction: slope < 0 ? "top-first" : slope > 0 ? "bottom-first" : "flat",
    ledRow,
    ledRowLagMs,
    midRowLagMs: at((height - 1) / 2),
    wholeFrameLagMs,
    offsetMs: wholeFrameLagMs - ledRowLagMs,
  };
}

/** Groups lags into clusters split wherever consecutive values differ by more than `gapMs`. */
export function clusterLags(lags: number[], gapMs: number): { centerMs: number; n: number }[] {
  const sorted = lags.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const out: { centerMs: number; n: number }[] = [];
  let start = 0;
  for (let i = 1; i <= sorted.length; i++) {
    if (i === sorted.length || sorted[i] - sorted[i - 1] > gapMs) {
      const group = sorted.slice(start, i);
      const sum = group.reduce((a, b) => a + b, 0);
      out.push({ centerMs: sum / group.length, n: group.length });
      start = i;
    }
  }
  return out;
}

// ---------------------------------------------------------------- exposure window

/**
 * Where the camera's integration window sits relative to `captureTime`.
 *
 * The free-running train fires short pulses at a phase uniformly random with respect to the
 * camera's frame clock. For each pulse: the first frame k that caught any of it, the fraction
 * f of the pulse's light that landed in frame k rather than the next one, and the phase of
 * the pulse relative to frame k's stamp. Sorted by phase, f is 1 for pulses that fell wholly
 * inside the window and falls off at both edges, so the 0.9 and 0.1 crossings bracket the
 * window and `windowMs` is the integration time.
 *
 * The normaliser is the two-frame total, as the pulse's light is split between frames k and
 * k+1 whenever consecutive exposures are contiguous — but a camera whose exposure is shorter
 * than its frame period drops the light that falls in the dead time altogether, and then the
 * two-frame total collapses onto frame k and f would be identically 1. So the denominator is
 * floored at the full LED level: identical where the exposures tile, and the actual captured
 * fraction where they do not.
 */
export function exposureWindow(
  leds: RigLed[],
  frames: CamFrame[],
  fit: SyncFit,
  opts: DetectOptions = {},
): ExposureResult {
  const detector = ledDetector(leds, frames, fit, opts);
  const sortedFrames = frames.slice().sort((a, b) => a.captureTime - b.captureTime);
  const before = opts.searchBeforeMs ?? DEFAULT_SEARCH_BEFORE_MS;
  const after = opts.searchAfterMs ?? DEFAULT_SEARCH_AFTER_MS;
  const train = leds.filter((l) => l.src === "T");
  const full = detector.ledLevel - detector.baseline;

  const period = frameInterval(sortedFrames);
  const points: ExposurePoint[] = [];
  if (detector.detected) {
    for (const led of train) {
      const tOnMs = toBrowserMs(fit, led.tOnUs);
      const pulseMs = toBrowserMs(fit, led.tOffUs) - tOnMs;
      const k = firstLitFrame(sortedFrames, tOnMs, detector.noiseThreshold, before, after);
      if (k < 0 || k + 1 >= sortedFrames.length) continue;
      const a = sortedFrames[k].peak - detector.baseline;
      const b = sortedFrames[k + 1].peak - detector.baseline;
      // A pulse longer than a frame lights the next frame fully, so the first frame's
      // fraction is simply its share of the full level. A pulse shorter than a frame is
      // split between at most two frames, and their total is the reference (floored at the
      // full level for a camera whose exposure does not tile the period).
      const denom = pulseMs >= period ? full : Math.max(a + b, full);
      if (!(denom > 0)) continue;
      points.push({ phaseMs: tOnMs - sortedFrames[k].captureTime, f: clamp01(a / denom) });
    }
  }
  points.sort((p, q) => p.phaseMs - q.phaseMs);

  // The onset lands at a uniformly random point in the exposure, so f is linear in phase
  // across the window: 1 where the LED came on before the exposure began, 0 where it came on
  // as the exposure ended. Fit the ramp and extrapolate to f = 1 and f = 0 rather than
  // looking for threshold crossings — the detector never returns a first-lit frame below
  // its noise floor, so the bottom of the ramp is cut off and a 0.1 crossing may never exist.
  const ramp = points.filter((p) => p.f >= RAMP_LO && p.f <= RAMP_HI);
  let startMs = NaN;
  let endMs = NaN;
  let r2 = NaN;
  if (ramp.length >= MIN_RAMP_POINTS) {
    const n = ramp.length;
    let sx = 0;
    let sy = 0;
    for (const p of ramp) {
      sx += p.phaseMs;
      sy += p.f;
    }
    const mx = sx / n;
    const my = sy / n;
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (const p of ramp) {
      sxx += (p.phaseMs - mx) * (p.phaseMs - mx);
      sxy += (p.phaseMs - mx) * (p.f - my);
      syy += (p.f - my) * (p.f - my);
    }
    if (sxx > 0 && syy > 0) {
      const b = sxy / sxx;
      const a = my - b * mx;
      r2 = (sxy * sxy) / (sxx * syy);
      if (b !== 0) {
        const at1 = (1 - a) / b;
        const at0 = -a / b;
        startMs = Math.min(at1, at0);
        endMs = Math.max(at1, at0);
      }
    }
  }
  return {
    n: points.length,
    points,
    startMs,
    endMs,
    windowMs: endMs - startMs,
    nRamp: ramp.length,
    r2,
    ledNotDetected: !detector.detected,
    detector,
  };
}
