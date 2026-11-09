// Provenance: identical to `packages/core/src/loopback.ts` in jspsych/saccadejs, which is
// the shipping copy. This one exists so the rig panel can run standalone against the
// firmware in ../firmware/. The rig is what validated the estimator, so if the two ever
// diverge, re-validate here first: `diff` them before trusting a rig measurement.
//
// Screen -> webcam timing loopback analysis.
//
// The browser can stamp a gaze sample with the camera frame's `captureTime`, but two
// physical lags stay invisible to JS: display lag (rAF flip -> photons leaving the panel)
// and camera lag (photons -> `captureTime`). Their sum is the offset needed to align gaze
// samples with stimulus onsets. This module recovers that sum by cross-correlating a known
// black/white flash sequence against the webcam's mean luminance.
//
// Pure / DOM-free on purpose: the whole thing runs under vitest's node environment.

export interface Flip {
  /** Flip time on the `performance.now()` timeline (rAF timestamp). */
  t: number;
  /** 0 = black, 1 = white. */
  level: number;
}

export interface LumSample {
  /** Sample time on the `performance.now()` timeline. */
  t: number;
  /** Mean luminance of the camera frame (arbitrary units). */
  lum: number;
}

export interface LagOptions {
  minLagMs?: number;
  maxLagMs?: number;
  stepMs?: number;
  /**
   * Minimum distance from the peak for a lag to count as a competing peak. Must exceed the
   * symbol duration, otherwise the shoulder of the main correlation lobe is reported as
   * ambiguity. Default SECOND_PEAK_MIN_SEPARATION_MS.
   */
  secondPeakSeparationMs?: number;
}

export interface CurvePoint {
  lagMs: number;
  r: number;
}

export interface LagResult {
  /** Raw argmax of the correlation curve. */
  lagMs: number;
  /** Parabolic refinement of the argmax over its two neighbours. */
  lagRefinedMs: number;
  /** Pearson r at the peak. */
  peakCorr: number;
  /** Best r at least 40 ms away from the peak, divided by `peakCorr`. Ambiguity measure. */
  secondPeakRatio: number;
  curve: CurvePoint[];
  /** Samples that contributed at the peak lag. */
  nSamples: number;
  /**
   * The contiguous run of lags around the peak with (numerically) the same r. With few,
   * widely spaced edges the curve is piecewise constant: r only changes when some sample
   * crosses a shifted edge, so the set of lags consistent with every edge is a flat top,
   * not a peak. Its center is the estimate and its width the uncertainty (sparse mode).
   * With a dense m-sequence the plateau collapses to about one step.
   */
  plateau: Plateau;
}

export interface Plateau {
  lowMs: number;
  highMs: number;
  centerMs: number;
  widthMs: number;
}

export interface IntervalStats {
  mean: number;
  sd: number;
  min: number;
  max: number;
  n: number;
}

export interface SplitHalves {
  first: LagResult;
  second: LagResult;
}

/** Minimum separation from the peak, in ms, for a candidate to count as a second peak. */
export const SECOND_PEAK_MIN_SEPARATION_MS = 40;

const DEFAULT_MIN_LAG_MS = 0;
const DEFAULT_MAX_LAG_MS = 600;
const DEFAULT_STEP_MS = 1;

/**
 * Galois-form LFSR feedback masks for primitive polynomials of each order — the classic
 * maximal-length tap sets. Orders 7 and 8 are what the timing page offers; the rest are
 * here so the module stays generally useful.
 */
const POLY: Record<number, number> = {
  3: 0x6,
  4: 0xc,
  5: 0x14,
  6: 0x30,
  7: 0x60,
  8: 0xb8,
  9: 0x110,
  10: 0x240,
  11: 0x500,
  12: 0xe08,
};

/**
 * Maximal-length binary LFSR (m-)sequence of length 2^order - 1.
 * Deterministic: the same order always yields the same bits.
 */
export function mSequence(order: number): Uint8Array {
  const poly = POLY[order];
  if (poly === undefined) throw new Error(`unsupported m-sequence order ${order}`);
  const n = (1 << order) - 1;
  const out = new Uint8Array(n);
  let state = n; // all-ones seed; any non-zero seed works
  for (let i = 0; i < n; i++) {
    const lsb = state & 1;
    out[i] = lsb;
    state >>= 1;
    if (lsb) state ^= poly;
  }
  return out;
}

/**
 * Step function: the level of the most recent flip at or before `t`.
 * Returns NaN before the first flip (callers skip those samples).
 * `flips` must be sorted ascending by `t`.
 */
export function stimulusAt(flips: Flip[], t: number): number {
  if (flips.length === 0 || t < flips[0].t) return NaN;
  let lo = 0;
  let hi = flips.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (flips[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return flips[lo].level;
}

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 3) return 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const den = Math.sqrt(sxx * syy);
  if (!(den > 0)) return 0;
  return sxy / den;
}

/**
 * Scan candidate lags tau and correlate luminance against the stimulus shifted back by tau.
 * Sub-frame precision comes from scanning tau continuously while the stimulus stays a step
 * function evaluated at the (irregular) camera sample times.
 */
export function estimateLag(flips: Flip[], samples: LumSample[], opts: LagOptions = {}): LagResult {
  const minLag = opts.minLagMs ?? DEFAULT_MIN_LAG_MS;
  const maxLag = opts.maxLagMs ?? DEFAULT_MAX_LAG_MS;
  const step = opts.stepMs ?? DEFAULT_STEP_MS;
  const separation = opts.secondPeakSeparationMs ?? SECOND_PEAK_MIN_SEPARATION_MS;
  if (!(step > 0)) throw new Error("stepMs must be > 0");
  if (maxLag < minLag) throw new Error("maxLagMs must be >= minLagMs");

  const sorted = flips.slice().sort((a, b) => a.t - b.t);
  const steps = Math.max(0, Math.round((maxLag - minLag) / step));

  const curve: CurvePoint[] = [];
  let bestIdx = 0;
  let bestR = -Infinity;
  let bestN = 0;

  const lum: number[] = [];
  const stim: number[] = [];
  for (let k = 0; k <= steps; k++) {
    const tau = minLag + k * step;
    lum.length = 0;
    stim.length = 0;
    for (const s of samples) {
      const v = stimulusAt(sorted, s.t - tau);
      if (!Number.isFinite(v)) continue;
      lum.push(s.lum);
      stim.push(v);
    }
    const r = pearson(lum, stim);
    curve.push({ lagMs: tau, r });
    if (r > bestR) {
      bestR = r;
      bestIdx = curve.length - 1;
      bestN = lum.length;
    }
  }

  if (curve.length === 0) {
    const plateau = { lowMs: NaN, highMs: NaN, centerMs: NaN, widthMs: NaN };
    return {
      lagMs: NaN,
      lagRefinedMs: NaN,
      peakCorr: 0,
      secondPeakRatio: 1,
      curve,
      nSamples: 0,
      plateau,
    };
  }

  const peak = curve[bestIdx];
  let second = -Infinity;
  for (const p of curve) {
    if (Math.abs(p.lagMs - peak.lagMs) < separation) continue;
    if (p.r > second) second = p.r;
  }
  const secondClamped = Number.isFinite(second) ? Math.max(0, second) : 0;
  const secondPeakRatio = peak.r > 0 ? secondClamped / peak.r : 1;

  const result: LagResult = {
    lagMs: peak.lagMs,
    lagRefinedMs: peak.lagMs,
    peakCorr: peak.r,
    secondPeakRatio,
    curve,
    nSamples: bestN,
    plateau: plateauAround(curve, bestIdx),
  };
  result.lagRefinedMs = refineLag(result);
  return result;
}

const PLATEAU_TOL = 1e-9;

/**
 * Edge-difference estimator for the sparse (no-flicker) schedule.
 *
 * Webcam auto-exposure re-adapts within a few hundred ms, so with 0.5–1 s holds the
 * luminance response to an edge is a jump followed by a decay back to baseline — not a
 * sustained step. Against that, the step-template correlation in `estimateLag` is flat for
 * every lag *below* the true one and biased low. This estimator uses only the jump: for
 * each edge e at t_e with direction d_e (+1 dark→light, −1 light→dark), and a candidate lag
 * τ, take the frame-to-frame luminance difference at the first sample at or after t_e + τ,
 * multiply by d_e, and sum over edges. D(τ) is piecewise constant and peaks exactly when
 * every edge's chosen sample is the one containing its jump, so the flat top is the set of
 * lags consistent with every edge and its center is the estimate. Nothing slower than one
 * frame (exposure loop, gain, drift, the subject moving) can affect it.
 *
 * `curve.r` holds D(τ) normalised to the sum of per-edge |jump| magnitudes, so 1.0 means
 * every edge's full jump was captured at that lag.
 */
export function estimateLagEdges(
  flips: Flip[],
  samples: LumSample[],
  opts: LagOptions = {},
): LagResult {
  const minLag = opts.minLagMs ?? DEFAULT_MIN_LAG_MS;
  const maxLag = opts.maxLagMs ?? DEFAULT_MAX_LAG_MS;
  const step = opts.stepMs ?? DEFAULT_STEP_MS;
  if (!(step > 0)) throw new Error("stepMs must be > 0");
  if (maxLag < minLag) throw new Error("maxLagMs must be >= minLagMs");

  const sortedFlips = flips.slice().sort((a, b) => a.t - b.t);
  const sorted = samples.slice().sort((a, b) => a.t - b.t);
  const empty = { lowMs: NaN, highMs: NaN, centerMs: NaN, widthMs: NaN };
  if (sortedFlips.length < 2 || sorted.length < 3) {
    return {
      lagMs: NaN,
      lagRefinedMs: NaN,
      peakCorr: 0,
      secondPeakRatio: 1,
      curve: [],
      nSamples: 0,
      plateau: empty,
    };
  }

  // Edges = level changes after the first flip (which only pins the initial level).
  const edges: { t: number; dir: number }[] = [];
  for (let i = 1; i < sortedFlips.length; i++) {
    const d = sortedFlips[i].level - sortedFlips[i - 1].level;
    if (d !== 0) edges.push({ t: sortedFlips[i].t, dir: Math.sign(d) });
  }
  const ts = sorted.map((s) => s.t);
  const diff = new Float64Array(sorted.length);
  for (let k = 1; k < sorted.length; k++) diff[k] = sorted[k].lum - sorted[k - 1].lum;

  // Normaliser: the largest |diff| within the lag window after each edge, summed.
  let scale = 0;
  for (const e of edges) {
    let best = 0;
    for (let k = lowerBound(ts, e.t + minLag); k < ts.length && ts[k] <= e.t + maxLag; k++)
      if (k > 0 && Math.abs(diff[k]) > best) best = Math.abs(diff[k]);
    scale += best;
  }
  if (!(scale > 0)) scale = 1;

  const steps = Math.max(0, Math.round((maxLag - minLag) / step));
  const curve: CurvePoint[] = [];
  let bestIdx = 0;
  let bestD = -Infinity;
  let bestN = 0;
  for (let i = 0; i <= steps; i++) {
    const tau = minLag + i * step;
    let d = 0;
    let n = 0;
    for (const e of edges) {
      const k = lowerBound(ts, e.t + tau);
      if (k <= 0 || k >= ts.length) continue;
      d += e.dir * diff[k];
      n++;
    }
    const r = d / scale;
    curve.push({ lagMs: tau, r });
    if (r > bestD) {
      bestD = r;
      bestIdx = curve.length - 1;
      bestN = n;
    }
  }
  const peak = curve[bestIdx];
  const result: LagResult = {
    lagMs: peak.lagMs,
    lagRefinedMs: peak.lagMs,
    peakCorr: peak.r,
    secondPeakRatio: 1,
    curve,
    nSamples: bestN,
    plateau: plateauAround(curve, bestIdx),
  };
  result.lagRefinedMs = result.plateau.centerMs;
  return result;
}

/** First index k with ts[k] >= t (ts ascending); ts.length if none. */
function lowerBound(ts: number[], t: number): number {
  let lo = 0;
  let hi = ts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Extent of the flat top containing `idx`: neighbours whose r equals the peak's within tolerance. */
function plateauAround(curve: CurvePoint[], idx: number): Plateau {
  const top = curve[idx].r;
  let lo = idx;
  let hi = idx;
  while (lo > 0 && Math.abs(curve[lo - 1].r - top) <= PLATEAU_TOL) lo--;
  while (hi < curve.length - 1 && Math.abs(curve[hi + 1].r - top) <= PLATEAU_TOL) hi++;
  const lowMs = curve[lo].lagMs;
  const highMs = curve[hi].lagMs;
  return { lowMs, highMs, centerMs: (lowMs + highMs) / 2, widthMs: highMs - lowMs };
}

/**
 * Transition times (ms from start) for the no-flicker mode: alternating levels held for a
 * random duration in [minGapMs, maxGapMs]. With 500–1000 ms gaps that is at most one flash
 * (a pair of opposing transitions) per second, a third of the WCAG 2.3.1 / Harding limit of
 * three per second. Timing information lives in the edges, not the rate, so ~20 edges over
 * 15 s localise the lag to a few ms once the camera's random sampling phase is intersected.
 * `rand` in [0, 1) — inject a seeded generator for reproducible runs.
 */
export function sparseSchedule(
  durationMs: number,
  minGapMs: number,
  maxGapMs: number,
  rand: () => number = Math.random,
): number[] {
  if (!(minGapMs > 0) || maxGapMs < minGapMs) throw new Error("bad gap range");
  const out: number[] = [];
  let t = minGapMs + (maxGapMs - minGapMs) * rand();
  while (t < durationMs) {
    out.push(t);
    t += minGapMs + (maxGapMs - minGapMs) * rand();
  }
  return out;
}

/** Small deterministic PRNG (mulberry32) so a schedule can be replayed from its seed. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Parabolic interpolation of the correlation peak over its immediate neighbours.
 * Falls back to the raw argmax at the curve edges or on a degenerate fit.
 */
export function refineLag(result: Pick<LagResult, "curve" | "lagMs"> & Partial<LagResult>): number {
  const { curve, lagMs } = result;
  if (curve.length < 3) return lagMs;
  const i = curve.findIndex((p) => p.lagMs === lagMs);
  if (i <= 0 || i >= curve.length - 1) return lagMs;
  const y0 = curve[i - 1].r;
  const y1 = curve[i].r;
  const y2 = curve[i + 1].r;
  const den = y0 - 2 * y1 + y2;
  if (!(Math.abs(den) > 1e-12)) return lagMs;
  const delta = (0.5 * (y0 - y2)) / den;
  if (!Number.isFinite(delta) || Math.abs(delta) > 1) return lagMs;
  const stepLeft = curve[i].lagMs - curve[i - 1].lagMs;
  const stepRight = curve[i + 1].lagMs - curve[i].lagMs;
  const stepMs = delta < 0 ? stepLeft : stepRight;
  return lagMs + delta * stepMs;
}

/** Mean / sd / min / max of the consecutive differences of `ts`. */
export function intervalStats(ts: number[]): IntervalStats {
  const d: number[] = [];
  for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
  const n = d.length;
  if (n === 0) return { mean: 0, sd: 0, min: 0, max: 0, n: 0 };
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of d) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / n;
  let ss = 0;
  for (const v of d) ss += (v - mean) * (v - mean);
  const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
  return { mean, sd, min, max, n };
}

/** Run `estimateLag` separately on the earlier and later half of the samples. */
export function splitHalves(
  flips: Flip[],
  samples: LumSample[],
  opts: LagOptions = {},
  estimator: (f: Flip[], s: LumSample[], o: LagOptions) => LagResult = estimateLag,
): SplitHalves {
  const sorted = samples.slice().sort((a, b) => a.t - b.t);
  const mid = Math.floor(sorted.length / 2);
  return {
    first: estimator(flips, sorted.slice(0, mid), opts),
    second: estimator(flips, sorted.slice(mid), opts),
  };
}
