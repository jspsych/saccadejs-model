import { describe, expect, it } from "vitest";
import {
  chooseLedSignal,
  displayLag,
  exposureWindow,
  fitSync,
  ledDetector,
  localiseLed,
  relayPlateau,
  relayResiduals,
  roiFrames,
  rowBands,
  toBrowserMs,
} from "../src/rigAnalysis";
import { stimulusAt } from "../src/loopback";
import type { CamFrame, CamFrameGray, Flip, RigEdge, RigLed, SyncPair } from "../src/rigAnalysis";

/** Deterministic PRNG so the synthetic cases never flake. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rnd: () => number): () => number {
  return () => {
    const u = Math.max(rnd(), 1e-12);
    const v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// The synthetic rig clock: browserMs = OFFSET + SLOPE * rigUs / 1000 (300 ppm fast).
const OFFSET = 1234.5;
const SLOPE = 1.0003;
const rigToBrowser = (rigUs: number): number => OFFSET + (SLOPE * rigUs) / 1000;
const browserToRig = (ms: number): number => ((ms - OFFSET) / SLOPE) * 1000;

/** Sync pings, ~1 ms RTT, +/- 0.4 ms of jitter on each side of the reply. */
function syncPings(n: number, seed: number, startMs = 0): SyncPair[] {
  const rnd = mulberry32(seed);
  const gauss = gaussian(mulberry32(seed ^ 0x5bf03635));
  const out: SyncPair[] = [];
  let t = startMs;
  for (let i = 0; i < n; i++) {
    const tSend = t;
    const outbound = 0.5 + 0.4 * gauss();
    const inbound = 0.5 + 0.4 * gauss();
    const tAnswer = tSend + Math.max(outbound, 0.05);
    const tRecv = tAnswer + Math.max(inbound, 0.05);
    out.push({ token: i + 1, tSend, tRecv, rigUs: browserToRig(tAnswer) });
    t += 20 + 40 * rnd();
  }
  return out;
}

/**
 * What the page actually collects: 20 pings before the run and 20 after, so the fitted slope
 * is pinned by the whole span instead of extrapolated out of a one-second window.
 */
function syncSpan(seed: number, endMs: number): SyncPair[] {
  return [...syncPings(20, seed), ...syncPings(20, seed ^ 0x1234, endMs)];
}

describe("fitSync", () => {
  it("recovers the offset and slope from jittered pings", () => {
    const fit = fitSync(syncPings(20, 4242));
    expect(fit.n).toBe(20);
    expect(Math.abs(fit.a - OFFSET)).toBeLessThan(0.3);
    expect(Math.abs(fit.b - 1)).toBeLessThan(2e-3);
    expect(fit.rttMedianMs).toBeGreaterThan(0.5);
    expect(fit.residualSdMs).toBeLessThan(1);
    // The mapping itself is what matters, and it is good across the ping window.
    for (const p of syncPings(8, 99)) {
      expect(Math.abs(toBrowserMs(fit, p.rigUs) - rigToBrowser(p.rigUs))).toBeLessThan(0.5);
    }
  });

  it("drops pings whose round trip blew out", () => {
    const pairs = syncPings(20, 7);
    const stall = pairs[10];
    pairs[10] = { ...stall, tRecv: stall.tRecv + 40 };
    const fit = fitSync(pairs);
    expect(fit.n).toBeGreaterThanOrEqual(17);
    expect(fit.n).toBeLessThan(20);
    expect(fit.rttMaxMs).toBeGreaterThan(40);
    expect(Math.abs(fit.a - OFFSET)).toBeLessThan(0.3);
  });

  it("degrades gracefully on empty input", () => {
    const fit = fitSync([]);
    expect(fit.n).toBe(0);
    expect(Number.isNaN(fit.a)).toBe(true);
  });
});

// ---------------------------------------------------------------- shared synthesis

const RISE_MS = 24;
const FALL_MS = 130;

/** Sparse alternating flips, gaps 500-1000 ms, starting at t = 1000. */
function sparseFlips(durationMs: number, seed: number): Flip[] {
  const rnd = mulberry32(seed);
  const flips: Flip[] = [{ t: 1000, level: 0 }];
  let t = 1000;
  let level = 0;
  while (t < 1000 + durationMs) {
    t += 500 + 500 * rnd();
    level = 1 - level;
    flips.push({ t, level });
  }
  return flips;
}

/** One photodiode edge per flip: 24 ms after a rise, 130 ms after a fall. */
function edgesFor(flips: Flip[]): RigEdge[] {
  const out: RigEdge[] = [];
  for (let i = 1; i < flips.length; i++) {
    const dir = flips[i].level === 1 ? "R" : "F";
    const ms = flips[i].t + (dir === "R" ? RISE_MS : FALL_MS);
    out.push({
      tUs: browserToRig(ms),
      dir,
      level: dir === "R" ? 3000 : 500,
      envMin: 400,
      envMax: 3100,
    });
  }
  return out;
}

const FIT = fitSync(syncSpan(3, 201000));

describe("displayLag", () => {
  it("recovers a 24 ms rising and 130 ms falling display lag", () => {
    const flips = sparseFlips(30000, 11);
    const r = displayLag(flips, edgesFor(flips), FIT);
    expect(r.rising.n).toBeGreaterThan(10);
    expect(r.falling.n).toBeGreaterThan(10);
    expect(Math.abs(r.rising.meanMs - RISE_MS)).toBeLessThan(1);
    expect(Math.abs(r.falling.meanMs - FALL_MS)).toBeLessThan(1);
    expect(r.rising.sdMs).toBeLessThan(0.5);
    expect(r.unmatchedFlips).toBe(0);
    expect(r.unmatchedEdges).toBe(0);
    expect(r.pairs.length).toBe(r.rising.n + r.falling.n);
  });

  it("counts flips with no edge, and edges with no flip", () => {
    const flips = sparseFlips(10000, 5);
    const edges = edgesFor(flips);
    const dropped = edges.filter((_, i) => i !== 2);
    const short = displayLag(flips, dropped, FIT);
    expect(short.unmatchedFlips).toBe(1);
    // A falling edge 130 ms out is beyond a 60 ms window, so only the rises match.
    const tight = displayLag(flips, edges, FIT, 60);
    expect(tight.falling.n).toBe(0);
    expect(tight.unmatchedEdges).toBe(edges.length - tight.rising.n);
  });
});

// ---------------------------------------------------------------- camera synthesis

interface CameraOptions {
  /** Exposure window relative to captureTime, e.g. [-20, 0]. */
  exposure: [number, number];
  periodMs: number;
  baseline: number;
  amplitude: number;
  noiseSd: number;
}

/**
 * Camera frames whose `peak` is the LED's light integrated over the frame's exposure window,
 * normalised so a frame lit for its whole exposure reads `baseline + amplitude`.
 */
function cameraFrames(
  startMs: number,
  endMs: number,
  windows: { onMs: number; offMs: number }[],
  opts: CameraOptions,
  seed: number,
  jitterMs = 3,
): CamFrame[] {
  const rnd = mulberry32(seed);
  const gauss = gaussian(mulberry32(seed ^ 0x27d4eb2d));
  const [expStart, expEnd] = opts.exposure;
  const expLen = expEnd - expStart;
  const out: CamFrame[] = [];
  let t = startMs;
  while (t <= endMs) {
    const lo = t + expStart;
    const hi = t + expEnd;
    let overlap = 0;
    for (const w of windows) {
      if (w.offMs < lo) continue;
      if (w.onMs > hi) break;
      overlap += Math.max(0, Math.min(w.offMs, hi) - Math.max(w.onMs, lo));
    }
    out.push({
      captureTime: t,
      now: t + 1.5,
      peak: opts.baseline + (opts.amplitude * overlap) / expLen + opts.noiseSd * gauss(),
    });
    t += opts.periodMs + (rnd() * 2 - 1) * jitterMs;
  }
  return out;
}

/** Relay one LED pulse off every rising edge, with a random sub-frame delay. */
function relayLeds(edges: RigEdge[], seed: number, widthUs = 100000): RigLed[] {
  const rnd = mulberry32(seed);
  return edges
    .filter((e) => e.dir === "R")
    .map((e) => {
      const delayUs = Math.round(40000 * rnd());
      return {
        tOnUs: e.tUs + delayUs,
        tOffUs: e.tUs + delayUs + widthUs,
        src: "R" as const,
        delayUs,
        edgeTUs: e.tUs,
      };
    });
}

const CAM: CameraOptions = {
  exposure: [-20, 0],
  periodMs: 1000 / 30,
  baseline: 30,
  amplitude: 60,
  noiseSd: 0.5,
};

/**
 * The lag the loopback should report for this camera: the display lag, plus the mean delay
 * from a photon to the `captureTime` of the first frame that clearly contains it — half the
 * exposure (the detector fires at 50 % of one exposure) plus half a frame of quantisation.
 */
const TRUE_L = RISE_MS + 10 + CAM.periodMs / 2;

function relayScene(seed: number): {
  flips: Flip[];
  edges: RigEdge[];
  leds: RigLed[];
  frames: CamFrame[];
} {
  const flips = sparseFlips(120000, seed);
  const edges = edgesFor(flips);
  const leds = relayLeds(edges, seed + 1);
  const windows = leds.map((l) => ({
    onMs: rigToBrowser(l.tOnUs),
    offMs: rigToBrowser(l.tOffUs),
  }));
  const frames = cameraFrames(900, 1000 + 121000, windows, CAM, seed + 2);
  return { flips, edges, leds, frames };
}

/**
 * A camera that integrates continuously — exposure equal to the frame period, the usual case
 * in dim light. Then every LED onset lands inside some frame's integration, so the plateau
 * (the intersection of the per-pulse brackets) is well defined and nearly every pulse agrees
 * with it. The lag it converges on is the display lag plus half a frame: the "largest jump"
 * rule switches to the following frame exactly when the onset passes the middle of a frame's
 * integration, which is what makes the bracket intersection collapse there.
 */
const CONT_CAM: CameraOptions = {
  exposure: [-(1000 / 30), 0],
  periodMs: 1000 / 30,
  baseline: 30,
  amplitude: 60,
  noiseSd: 0.5,
};

function continuousRelayScene(seed: number): {
  flips: Flip[];
  edges: RigEdge[];
  leds: RigLed[];
  frames: CamFrame[];
} {
  const flips = sparseFlips(120000, seed);
  const edges = edgesFor(flips);
  const leds = relayLeds(edges, seed + 1);
  const windows = leds.map((l) => ({
    onMs: rigToBrowser(l.tOnUs),
    offMs: rigToBrowser(l.tOffUs),
  }));
  // No frame-interval jitter: a rolling grid would smear the switch point that defines the
  // plateau, and the point here is that the estimator finds it, not that it survives jitter.
  const frames = cameraFrames(900, 1000 + 121000, windows, { ...CONT_CAM }, seed + 2, 0);
  return { flips, edges, leds, frames };
}

describe("relayPlateau", () => {
  it("recovers the lag every pulse agrees with", () => {
    const { flips, edges, leds, frames } = continuousRelayScene(41);
    const r = relayResiduals(leds, edges, flips, frames, FIT, 0);
    const p = r.plateau;
    const truth = RISE_MS + CONT_CAM.periodMs / 2;
    expect(p.n).toBeGreaterThan(50);
    expect(Math.abs(p.centerMs - truth)).toBeLessThan(2);
    expect(p.consistent / p.n).toBeGreaterThanOrEqual(0.9);
    expect(p.lowMs).toBeLessThanOrEqual(p.centerMs);
    expect(p.highMs).toBeGreaterThanOrEqual(p.centerMs);
  });

  it("estimates the exposure from how often the first lit frame jumped the full amount", () => {
    const { flips, edges, leds, frames } = continuousRelayScene(43);
    const r = relayResiduals(leds, edges, flips, frames, FIT, 0);
    // Exposure == the frame period here, so essentially every onset splits across two
    // frames and the estimate should land near a full frame.
    expect(r.exposureEstimateMs).toBeGreaterThan(0.5 * CONT_CAM.periodMs);
    expect(r.framePeriodMs).toBeCloseTo(CONT_CAM.periodMs, 0);

    // The dead-time camera is the other extreme: most onsets fall between exposures, are
    // caught whole by the next frame, and the estimate drops towards the 20 ms exposure.
    const dead = relayScene(45);
    const rd = relayResiduals(dead.leds, dead.edges, dead.flips, dead.frames, FIT, 0);
    expect(rd.exposureEstimateMs).toBeLessThan(r.exposureEstimateMs);
  });

  it("returns NaN rather than a lag when nothing matched", () => {
    const p = relayPlateau([]);
    expect(p.n).toBe(0);
    expect(Number.isNaN(p.centerMs)).toBe(true);
  });
});

describe("ledDetector", () => {
  it("separates baseline from LED level and flags an LED that never showed up", () => {
    const { leds, frames } = relayScene(21);
    const d = ledDetector(leds, frames, FIT);
    expect(d.detected).toBe(true);
    expect(Math.abs(d.baseline - CAM.baseline)).toBeLessThan(1);
    expect(Math.abs(d.ledLevel - (CAM.baseline + CAM.amplitude))).toBeLessThan(2);
    expect(d.threshold).toBeGreaterThan(d.baseline);
    expect(d.threshold).toBeLessThan(d.ledLevel);

    const dark = frames.map((f) => ({ ...f, peak: CAM.baseline + 0.5 * (f.peak - f.peak) }));
    expect(ledDetector(leds, dark, FIT).detected).toBe(false);
  });
});

describe("relayResiduals", () => {
  it("is zero-mean when the loopback lag is right", () => {
    const { flips, edges, leds, frames } = relayScene(31);
    const r = relayResiduals(leds, edges, flips, frames, FIT, TRUE_L);
    expect(r.ledNotDetected).toBe(false);
    expect(r.n).toBeGreaterThan(50);
    expect(r.unmatched).toBe(0);
    expect(Math.abs(r.meanResidualMs)).toBeLessThan(3);
    // The spread is the frame quantisation: sd of uniform(0, 33.3) is 9.6 ms.
    expect(r.sdResidualMs).toBeLessThan(12);
    expect(Math.abs(r.meanDirectLagMs - TRUE_L)).toBeLessThan(3);
    expect(r.residuals.every((x) => x.dir === "R")).toBe(true);
    // The chosen frame is the one with the biggest rise, so it always holds at least half
    // the LED's light; f says how much, and the jump cleared the noise gate to be picked.
    expect(r.residuals.every((x) => x.f >= 0.4 && x.f <= 1)).toBe(true);
    expect(r.residuals.every((x) => x.jump > 0)).toBe(true);
    // Delays really were spread over the requested 0-40 ms.
    expect(Math.max(...r.residuals.map((x) => x.delayMs))).toBeGreaterThan(30);
  });

  it("reports a mean offset of -15 ms when the loopback lag is 15 ms too large", () => {
    const { flips, edges, leds, frames } = relayScene(31);
    const r = relayResiduals(leds, edges, flips, frames, FIT, TRUE_L + 15);
    expect(Math.abs(r.meanResidualMs + 15)).toBeLessThan(3);
    expect(Math.abs(r.meanDirectLagMs - TRUE_L)).toBeLessThan(3);
  });

  it("flags an undetectable LED instead of inventing residuals", () => {
    const { flips, edges, leds, frames } = relayScene(31);
    const flat = frames.map((f) => ({ ...f, captureTime: f.captureTime, peak: CAM.baseline }));
    const r = relayResiduals(leds, edges, flips, flat, FIT, TRUE_L);
    expect(r.ledNotDetected).toBe(true);
    expect(r.n).toBe(0);
    expect(r.unmatched).toBe(leds.length);
  });
});

// ---------------------------------------------------------------- ROI localisation

const ROI_W = 160;
const ROI_H = 120;
const ROI_PIX = ROI_W * ROI_H;
const PEAK_N = 24;

/** The brightest-N mean the page computes per frame, mirrored here. */
function brightestMean(plane: Uint8Array): number {
  const hist = new Int32Array(256);
  for (let i = 0; i < plane.length; i++) hist[plane[i]]++;
  let need = PEAK_N;
  let total = 0;
  let got = 0;
  for (let b = 255; b >= 0 && need > 0; b--) {
    const take = hist[b] <= need ? hist[b] : need;
    if (take === 0) continue;
    total += b * take;
    got += take;
    need -= take;
  }
  return got > 0 ? total / got : 0;
}

/** The page's plane reduction, mirrored here: gray, and redness = clamp(R - max(G, B)). */
function paintPlanes(
  rgb: (i: number, lit: boolean) => [number, number, number],
  lit: boolean,
  gauss: () => number,
  noiseSd: number,
): { gray: Uint8Array; red: Uint8Array } {
  const gray = new Uint8Array(ROI_PIX);
  const red = new Uint8Array(ROI_PIX);
  for (let i = 0; i < ROI_PIX; i++) {
    const [r, g, b] = rgb(i, lit);
    const n = noiseSd * gauss();
    const y = 0.299 * r + 0.587 * g + 0.114 * b + n;
    gray[i] = y < 0 ? 0 : y > 255 ? 255 : (y + 0.5) | 0;
    const rn = r - (g > b ? g : b) + n;
    red[i] = rn < 0 ? 0 : rn > 255 ? 255 : (rn + 0.5) | 0;
  }
  return { gray, red };
}

/** 3x3 blob centred on (bx, by). */
function blobIndices(bx: number, by: number): Set<number> {
  const out = new Set<number>();
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) out.add((by + dy) * ROI_W + bx + dx);
  return out;
}

function ledWindowsOf(leds: RigLed[]): { onMs: number; offMs: number }[] {
  return leds.map((l) => ({ onMs: rigToBrowser(l.tOnUs), offMs: rigToBrowser(l.tOffUs) }));
}

function sceneFrames(
  leds: RigLed[],
  rgb: (i: number, lit: boolean) => [number, number, number],
  seed: number,
  endMs: number,
  periodMs = 1000 / 30,
): CamFrameGray[] {
  const rnd = mulberry32(seed);
  const gauss = gaussian(mulberry32(seed ^ 0x85ebca6b));
  const windows = ledWindowsOf(leds);
  const out: CamFrameGray[] = [];
  let t = 1000;
  while (t <= endMs) {
    const lit = windows.some((w) => t >= w.onMs && t <= w.offMs);
    const { gray, red } = paintPlanes(rgb, lit, gauss, 1.5);
    out.push({
      captureTime: t,
      now: t + 1.5,
      peak: brightestMean(gray),
      peakRed: brightestMean(red),
      gray,
      red,
    });
    t += periodMs + (rnd() * 2 - 1) * 3;
  }
  return out;
}

/** A train of `n` pulses of `widthMs`, one per second, like the panel's Locate LED (400 ms) or Test 3 (100 ms). */
function trainPulses(n: number, startMs: number, widthMs: number): RigLed[] {
  const out: RigLed[] = [];
  for (let i = 0; i < n; i++) {
    const on = startMs + i * 1000 + ((i * 137) % 400);
    out.push({
      tOnUs: browserToRig(on),
      tOffUs: browserToRig(on + widthMs),
      src: "T",
      delayUs: 0,
      edgeTUs: 0,
    });
  }
  return out;
}

/** A train of `n` 400 ms pulses 600 ms apart, like the panel's Locate LED. */
function locatePulses(n: number, startMs: number): RigLed[] {
  const out: RigLed[] = [];
  for (let i = 0; i < n; i++) {
    const on = startMs + i * 1000;
    out.push({
      tOnUs: browserToRig(on),
      tOffUs: browserToRig(on + 400),
      src: "T",
      delayUs: 0,
      edgeTUs: 0,
    });
  }
  return out;
}

function separationOf(frames: CamFrame[], leds: RigLed[]): number {
  const d = ledDetector(leds, frames, FIT);
  return d.ledLevel - d.baseline;
}

describe("localiseLed", () => {
  const BX = 120;
  const BY = 40;
  const blob = blobIndices(BX, BY);
  // 22 static pixels brighter than anything else, so the LED never enters the top 24, and a
  // second tier that does — the situation a real face + desk lamp produces.
  const lamp = new Set<number>();
  for (let i = 0; i < 22; i++) lamp.add(10 * ROI_W + 10 + i);
  const tier = new Set<number>();
  for (let i = 0; i < 60; i++) tier.add(60 * ROI_W + 10 + i);

  it("finds a gray blob the brightest-24 detector cannot see", () => {
    const leds = locatePulses(6, 1500);
    const frames = sceneFrames(
      leds,
      (i, lit) => {
        if (lamp.has(i)) return [250, 250, 250];
        if (tier.has(i)) return [150, 150, 150];
        if (blob.has(i)) return lit ? [200, 200, 200] : [60, 60, 60];
        return [40, 40, 40];
      },
      7,
      8000,
    );

    const roi = localiseLed(frames, leds, FIT, ROI_W, ROI_H, { plane: "gray" });
    expect(roi).not.toBeNull();
    if (!roi) return;
    expect(roi.n).toBe(9);
    expect(roi.indices.every((i) => blob.has(i))).toBe(true);
    expect(roi.centroid.x).toBeCloseTo(BX, 5);
    expect(roi.centroid.y).toBeCloseTo(BY, 5);

    // The whole point: the ROI signal separates far better than the brightest-24 mean.
    const sepRoi = separationOf(roiFrames(frames, roi), leds);
    const sep24 = separationOf(
      frames.map((f) => ({ captureTime: f.captureTime, now: f.now, peak: f.peak })),
      leds,
    );
    expect(sep24).toBeGreaterThan(0);
    expect(sepRoi).toBeGreaterThanOrEqual(10 * sep24);
  });

  it("picks the redness plane for a red LED on a face with a white highlight", () => {
    const leds = locatePulses(6, 1500);
    const frames = sceneFrames(
      leds,
      (i, lit) => {
        if (lamp.has(i)) return [250, 250, 250];
        // Face: R > G > B but nowhere near saturated.
        if (blob.has(i) && lit) return [255, 60, 60];
        return [150, 120, 100];
      },
      9,
      8000,
    );

    const choice = chooseLedSignal(frames, leds, FIT, ROI_W, ROI_H);
    expect(choice.source).toBe("roi-red");
    expect(choice.roi?.plane).toBe("red");
    expect(choice.roi?.n).toBe(9);
    expect(choice.roi?.indices.every((i) => blob.has(i))).toBe(true);
    expect(choice.roi?.centroid.x).toBeCloseTo(BX, 5);
    // The LED is *darker* than the face in gray, so gray cannot see it at all.
    expect(choice.roi?.signalMads ?? 0).toBeGreaterThan(10 * (choice.roiGray?.signalMads ?? 0));
  });

  it("returns null rather than a noise ROI when the LED was never lit", () => {
    const leds = locatePulses(6, 1500);
    const frames = sceneFrames(leds, () => [40, 40, 40], 11, 6000);
    expect(localiseLed(frames, leds, FIT, ROI_W, ROI_H, { plane: "red" })).toBeNull();
    // With no LED at all the fallback is the redness brightest-24, not an invented ROI.
    const flat = sceneFrames([], () => [40, 40, 40], 12, 4000);
    expect(chooseLedSignal(flat, [], FIT, ROI_W, ROI_H).source).toBe("peak24-red");
  });
});

// ---------------------------------------------------------------- rolling shutter

/**
 * A camera with a rolling shutter: every row carries the same `captureTime`, but row r was
 * actually exposed `readoutMs * r / (H - 1)` later than the top row, so the screen edge
 * reaches the lower rows in an *earlier* frame and their apparent lag is smaller.
 * `peak`/`peakRed` are irrelevant here — only the gray plane is read.
 */
function rollingShutterFrames(
  flips: Flip[],
  baseLagMs: number,
  readoutMs: number,
  endMs: number,
  seed: number,
  nBands = 4,
): CamFrameGray[] {
  const rnd = mulberry32(seed);
  const gauss = gaussian(mulberry32(seed ^ 0x1b873593));
  const rowsPer = ROI_H / nBands;
  const out: CamFrameGray[] = [];
  let t = 1000;
  while (t <= endMs) {
    const gray = new Uint8Array(ROI_PIX);
    for (let b = 0; b < nBands; b++) {
      const row = b * rowsPer + (rowsPer - 1) / 2;
      const lag = baseLagMs - readoutMs * (row / (ROI_H - 1));
      const level = stimulusAt(flips, t - lag);
      const v = 40 + 150 * (Number.isFinite(level) ? level : 0) + 2 * gauss();
      gray.fill(
        v < 0 ? 0 : v > 255 ? 255 : (v + 0.5) | 0,
        b * rowsPer * ROI_W,
        (b + 1) * rowsPer * ROI_W,
      );
    }
    out.push({ captureTime: t, now: t + 1.5, peak: 0, peakRed: 0, gray, red: new Uint8Array(0) });
    t += 1000 / 30 + (rnd() * 2 - 1) * 3;
  }
  return out;
}

describe("localiseLed with short pulses and a slow camera (the Test 3 failure)", () => {
  const blob = blobIndices(105, 50);
  const scene = (i: number, lit: boolean): [number, number, number] => {
    if (blob.has(i)) return lit ? [230, 60, 60] : [120, 90, 80];
    return [130, 100, 90];
  };

  for (const [periodMs, minPerPulse] of [
    [43, 0.8],
    [1000 / 30, 1.5],
  ] as const) {
    it(`100 ms pulses at a ${periodMs.toFixed(1)} ms frame period still localise the LED`, () => {
      const leds = trainPulses(40, 1500, 100);
      const frames = sceneFrames(leds, scene, 11, 43000, periodMs);
      const roi = localiseLed(frames, leds, FIT, ROI_W, ROI_H, { plane: "red" });
      expect(roi).not.toBeNull();
      if (!roi) return;
      expect(roi.indices.every((i) => blob.has(i))).toBe(true);
      // Only the start of a pulse needs guarding (captureTime is the end of the exposure),
      // so a 100 ms pulse must still yield about one fully lit frame per pulse.
      expect(roi.nOn).toBeGreaterThanOrEqual(minPerPulse * leds.length);
    });
  }
});

describe("rowBands", () => {
  const READOUT_MS = 20;
  const BASE_LAG_MS = 60;

  it("recovers a 20 ms top-to-bottom readout", () => {
    const flips = sparseFlips(40000, 17);
    const frames = rollingShutterFrames(flips, BASE_LAG_MS, READOUT_MS, 41000, 5);
    const r = rowBands(frames, flips, ROI_W, ROI_H, 70, 45);
    expect(r.bands.length).toBe(4);
    expect(r.bands.every((b) => Number.isFinite(b.lagMs))).toBe(true);
    expect(Math.abs(r.readoutMs - READOUT_MS)).toBeLessThan(2);
    expect(r.direction).toBe("top-first");
    // Lower rows are read later, so their fitted lag is smaller.
    expect(r.bands[0].lagMs).toBeGreaterThan(r.bands[3].lagMs);
    // The fit at the LED's row lands between the outer bands, and the reported offset is
    // just the whole-frame lag minus it.
    expect(r.ledRowLagMs).toBeLessThan(r.bands[0].lagMs);
    expect(r.ledRowLagMs).toBeGreaterThan(r.bands[3].lagMs);
    expect(r.offsetMs).toBeCloseTo(70 - r.ledRowLagMs, 6);
  });

  it("reports no readout for a global shutter", () => {
    const flips = sparseFlips(40000, 19);
    const frames = rollingShutterFrames(flips, BASE_LAG_MS, 0, 41000, 6);
    const r = rowBands(frames, flips, ROI_W, ROI_H, 70, 45);
    expect(r.readoutMs).toBeLessThan(2);
    for (const b of r.bands) expect(Math.abs(b.lagMs - BASE_LAG_MS)).toBeLessThan(3);
  });

  it("degrades gracefully with no frames", () => {
    const r = rowBands([], sparseFlips(5000, 1), ROI_W, ROI_H, 70, null);
    expect(r.bands.length).toBe(0);
    expect(Number.isNaN(r.readoutMs)).toBe(true);
  });
});

describe("exposureWindow", () => {
  it("recovers a 12 ms exposure starting 20 ms before captureTime", () => {
    const rnd = mulberry32(808);
    // Free-running train: 100 ms pulses (Test 3's width) with random 600-1200 ms gaps.
    const leds: RigLed[] = [];
    let ms = 1500;
    while (ms < 200000) {
      const tOnUs = browserToRig(ms);
      leds.push({
        tOnUs,
        tOffUs: browserToRig(ms + 100),
        src: "T",
        delayUs: 0,
        edgeTUs: 0,
      });
      ms += 600 + 600 * rnd();
    }
    const windows = leds.map((l) => ({
      onMs: rigToBrowser(l.tOnUs),
      offMs: rigToBrowser(l.tOffUs),
    }));
    // Exposure 12 ms wide, ending 8 ms before the stamp: shorter than the frame period, so
    // pulses landing in the dead time are never seen at all.
    const frames = cameraFrames(1000, 201000, windows, { ...CAM, exposure: [-20, -8] }, 55);

    const r = exposureWindow(leds, frames, FIT);
    expect(r.ledNotDetected).toBe(false);
    expect(r.n).toBeGreaterThan(50);
    expect(Math.abs(r.startMs + 20)).toBeLessThan(2);
    expect(Math.abs(r.windowMs - 12)).toBeLessThan(2);
    expect(r.endMs).toBeGreaterThan(r.startMs);
    // Sorted by phase, and every f is a fraction.
    for (let i = 1; i < r.points.length; i++)
      expect(r.points[i].phaseMs).toBeGreaterThanOrEqual(r.points[i - 1].phaseMs);
    for (const p of r.points) {
      expect(p.f).toBeGreaterThanOrEqual(0);
      expect(p.f).toBeLessThanOrEqual(1);
    }
  });

  it("returns NaN edges rather than guessing when there is no train", () => {
    const { leds, frames } = relayScene(31);
    const r = exposureWindow(leds, frames, FIT);
    expect(r.n).toBe(0);
    expect(Number.isNaN(r.startMs)).toBe(true);
    expect(Number.isNaN(r.windowMs)).toBe(true);
  });
});
