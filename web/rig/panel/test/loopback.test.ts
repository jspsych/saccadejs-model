import { describe, expect, it } from "vitest";
import {
  estimateLag,
  estimateLagEdges,
  intervalStats,
  mSequence,
  refineLag,
  seededRandom,
  sparseSchedule,
  splitHalves,
  stimulusAt,
} from "../src/loopback";
import type { Flip, LumSample } from "../src/loopback";

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

const SYMBOL_MS = 66.7;
const SEQ_START_MS = 0;

function buildFlips(seq: Uint8Array, symbolMs = SYMBOL_MS): Flip[] {
  const flips: Flip[] = [];
  let prev = -1;
  for (let i = 0; i < seq.length; i++) {
    const level = seq[i];
    if (level !== prev) {
      flips.push({ t: SEQ_START_MS + i * symbolMs, level });
      prev = level;
    }
  }
  return flips;
}

/** 30 fps camera with +/- 3 ms interval jitter. */
function cameraTimes(startMs: number, endMs: number, rnd: () => number): number[] {
  const ts: number[] = [];
  let t = startMs;
  while (t <= endMs) {
    ts.push(t);
    t += 1000 / 30 + (rnd() * 2 - 1) * 3;
  }
  return ts;
}

function simulate(flips: Flip[], lagMs: number, seed: number, endMs = 8400): LumSample[] {
  const rnd = mulberry32(seed);
  const gauss = gaussian(mulberry32(seed ^ 0x9e3779b9));
  const ts = cameraTimes(700, endMs, rnd);
  return ts.map((t) => {
    const s = stimulusAt(flips, t - lagMs);
    return { t, lum: 40 + 120 * (Number.isFinite(s) ? s : 0) + 8 * gauss() };
  });
}

describe("mSequence", () => {
  for (const order of [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) {
    it(`order ${order} is maximal length and balanced`, () => {
      const seq = mSequence(order);
      const n = (1 << order) - 1;
      expect(seq.length).toBe(n);
      let ones = 0;
      for (const b of seq) {
        expect(b === 0 || b === 1).toBe(true);
        ones += b;
      }
      // A maximal-length LFSR emits exactly 2^(order-1) ones.
      expect(ones).toBe(1 << (order - 1));
    });
  }

  it("has no all-zero window of length `order` (circular)", () => {
    for (const order of [7, 8]) {
      const seq = mSequence(order);
      const n = seq.length;
      for (let i = 0; i < n; i++) {
        let zeros = 0;
        for (let k = 0; k < order; k++) zeros += seq[(i + k) % n] === 0 ? 1 : 0;
        expect(zeros).toBeLessThan(order);
      }
    }
  });

  it("is deterministic", () => {
    expect(Array.from(mSequence(7))).toEqual(Array.from(mSequence(7)));
  });

  it("rejects unsupported orders", () => {
    expect(() => mSequence(1)).toThrow();
  });
});

describe("stimulusAt", () => {
  const flips: Flip[] = [
    { t: 0, level: 1 },
    { t: 100, level: 0 },
    { t: 250, level: 1 },
  ];

  it("returns NaN before the first flip", () => {
    expect(Number.isNaN(stimulusAt(flips, -1))).toBe(true);
    expect(Number.isNaN(stimulusAt([], 5))).toBe(true);
  });

  it("holds the most recent level", () => {
    expect(stimulusAt(flips, 0)).toBe(1);
    expect(stimulusAt(flips, 99.9)).toBe(1);
    expect(stimulusAt(flips, 100)).toBe(0);
    expect(stimulusAt(flips, 249.9)).toBe(0);
    expect(stimulusAt(flips, 1e6)).toBe(1);
  });
});

describe("estimateLag synthetic recovery", () => {
  const flips = buildFlips(mSequence(7));

  it("recovers a true lag of 87 ms", () => {
    const samples = simulate(flips, 87, 12345);
    const r = estimateLag(flips, samples);
    expect(r.nSamples).toBeGreaterThan(150);
    expect(r.peakCorr).toBeGreaterThan(0.8);
    expect(Math.abs(r.lagRefinedMs - 87)).toBeLessThan(4);
    expect(r.secondPeakRatio).toBeLessThan(0.7);
  });

  it("recovers a true lag of 250 ms", () => {
    const samples = simulate(flips, 250, 987);
    const r = estimateLag(flips, samples);
    expect(r.peakCorr).toBeGreaterThan(0.8);
    expect(Math.abs(r.lagRefinedMs - 250)).toBeLessThan(4);
    expect(r.secondPeakRatio).toBeLessThan(0.7);
  });

  it("gives a weak peak for a camera signal uncorrelated with the stimulus", () => {
    const gauss = gaussian(mulberry32(4242));
    const rnd = mulberry32(24);
    const samples: LumSample[] = cameraTimes(700, 8400, rnd).map((t) => ({
      t,
      lum: 120 + 30 * gauss(),
    }));
    const r = estimateLag(flips, samples);
    expect(r.peakCorr).toBeLessThan(0.3);
  });

  it("honours the scan window options", () => {
    const samples = simulate(flips, 87, 5);
    const r = estimateLag(flips, samples, { minLagMs: 50, maxLagMs: 120, stepMs: 2 });
    expect(r.curve[0].lagMs).toBe(50);
    expect(r.curve[r.curve.length - 1].lagMs).toBe(120);
    expect(r.curve.length).toBe(36);
    expect(Math.abs(r.lagMs - 87)).toBeLessThanOrEqual(2);
  });
});

describe("refineLag", () => {
  it("interpolates a symmetric peak to its true centre", () => {
    const curve = [
      { lagMs: 9, r: 0.8 },
      { lagMs: 10, r: 1.0 },
      { lagMs: 11, r: 0.8 },
    ];
    const refined = refineLag({
      lagMs: 10,
      lagRefinedMs: 10,
      peakCorr: 1,
      secondPeakRatio: 0,
      curve,
      nSamples: 10,
    });
    expect(refined).toBeCloseTo(10, 6);
  });

  it("falls back to the raw argmax at the curve edge", () => {
    const curve = [
      { lagMs: 0, r: 1.0 },
      { lagMs: 1, r: 0.5 },
    ];
    const refined = refineLag({
      lagMs: 0,
      lagRefinedMs: 0,
      peakCorr: 1,
      secondPeakRatio: 0,
      curve,
      nSamples: 10,
    });
    expect(refined).toBe(0);
  });
});

describe("intervalStats", () => {
  it("summarises consecutive differences", () => {
    const s = intervalStats([0, 10, 20, 35]);
    expect(s.n).toBe(3);
    expect(s.mean).toBeCloseTo(35 / 3, 10);
    expect(s.min).toBe(10);
    expect(s.max).toBe(15);
    expect(s.sd).toBeGreaterThan(0);
  });

  it("handles degenerate input", () => {
    expect(intervalStats([]).n).toBe(0);
    expect(intervalStats([5]).n).toBe(0);
    expect(intervalStats([5, 15]).sd).toBe(0);
  });
});

describe("splitHalves", () => {
  it("both halves agree on the true lag", () => {
    const flips = buildFlips(mSequence(7));
    const samples = simulate(flips, 87, 777);
    const { first, second } = splitHalves(flips, samples);
    expect(Math.abs(first.lagRefinedMs - 87)).toBeLessThan(8);
    expect(Math.abs(second.lagRefinedMs - 87)).toBeLessThan(8);
    expect(Math.abs(first.lagRefinedMs - second.lagRefinedMs)).toBeLessThan(10);
  });
});

describe("sparse (no-flicker) mode", () => {
  function sparseFlips(durationMs: number, seed: number): Flip[] {
    const times = sparseSchedule(durationMs, 500, 1000, seededRandom(seed));
    const flips: Flip[] = [{ t: 0, level: 0 }];
    let level = 0;
    for (const t of times) {
      level = 1 - level;
      flips.push({ t, level });
    }
    return flips;
  }

  it("never exceeds one flash per second and yields ~20 edges in 15 s", () => {
    const times = sparseSchedule(15000, 500, 1000, seededRandom(1));
    expect(times.length).toBeGreaterThanOrEqual(15);
    expect(times.length).toBeLessThanOrEqual(30);
    for (let i = 1; i < times.length; i++)
      expect(times[i] - times[i - 1]).toBeGreaterThanOrEqual(500);
  });

  it("recovers an 87 ms lag from ~20 sparse edges via the plateau center", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const flips = sparseFlips(15000, seed);
      const samples = simulate(flips, 87, seed + 100, 15000);
      const r = estimateLag(flips, samples);
      expect(r.peakCorr).toBeGreaterThan(0.8);
      expect(r.plateau.widthMs).toBeLessThan(34);
      expect(Math.abs(r.plateau.centerMs - 87)).toBeLessThan(4);
    }
  });

  it("recovers a 250 ms lag", () => {
    const flips = sparseFlips(15000, 9);
    const samples = simulate(flips, 250, 42, 15000);
    const r = estimateLag(flips, samples);
    expect(Math.abs(r.plateau.centerMs - 250)).toBeLessThan(4);
  });

  it("plateau collapses to a few ms for a dense m-sequence", () => {
    const flips = buildFlips(mSequence(7), 83.3);
    const samples = simulate(flips, 87, 12345);
    const r = estimateLag(flips, samples);
    expect(r.plateau.widthMs).toBeLessThanOrEqual(5);
  });
});

/**
 * Webcam with auto-exposure: the response to a step is the step high-passed with time
 * constant T, i.e. a jump that decays back to baseline within a few hundred ms.
 */
function simulateAutoExposure(
  flips: Flip[],
  lagMs: number,
  seed: number,
  endMs: number,
  tauMs = 250,
): LumSample[] {
  const rnd = mulberry32(seed);
  const gauss = gaussian(mulberry32(seed ^ 0x9e3779b9));
  const ts = cameraTimes(700, endMs, rnd);
  let adapted = 0; // the exposure loop's estimate of the scene level
  let prevT = ts[0];
  return ts.map((t) => {
    const s0 = stimulusAt(flips, t - lagMs);
    const s = Number.isFinite(s0) ? s0 : 0;
    const dt = t - prevT;
    prevT = t;
    adapted += (s - adapted) * (1 - Math.exp(-dt / tauMs));
    return { t, lum: 100 + 120 * (s - adapted) + 6 * gauss() };
  });
}

describe("edge-difference estimator under auto-exposure", () => {
  function sparseFlips(durationMs: number, seed: number): Flip[] {
    const times = sparseSchedule(durationMs, 500, 1000, seededRandom(seed));
    const flips: Flip[] = [{ t: 0, level: 0 }];
    let level = 0;
    for (const t of times) {
      level = 1 - level;
      flips.push({ t, level });
    }
    return flips;
  }

  it("step-template correlation is not trustworthy when exposure adapts (documents the failure)", () => {
    const flips = sparseFlips(15000, 7);
    const samples = simulateAutoExposure(flips, 87, 77, 15000);
    const r = estimateLag(flips, samples);
    expect(r.plateau.centerMs < 80 || r.plateau.widthMs > 34 || r.peakCorr < 0.8).toBe(true);
  });

  it("edge estimator recovers 87 ms within 4 ms under auto-exposure", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const flips = sparseFlips(15000, seed);
      const samples = simulateAutoExposure(flips, 87, seed + 200, 15000);
      const r = estimateLagEdges(flips, samples);
      expect(r.plateau.widthMs).toBeLessThan(34);
      expect(Math.abs(r.plateau.centerMs - 87)).toBeLessThan(4);
      expect(r.peakCorr).toBeGreaterThan(0.7);
    }
  });

  it("edge estimator also works without auto-exposure and at 250 ms", () => {
    const flips = sparseFlips(15000, 11);
    const a = estimateLagEdges(flips, simulate(flips, 87, 5, 15000));
    expect(Math.abs(a.plateau.centerMs - 87)).toBeLessThan(4);
    const b = estimateLagEdges(flips, simulateAutoExposure(flips, 250, 6, 15000));
    expect(Math.abs(b.plateau.centerMs - 250)).toBeLessThan(4);
  });

  it("edge estimator on an m-sequence agrees with the correlation estimator", () => {
    const flips = buildFlips(mSequence(7), 83.3);
    const samples = simulate(flips, 87, 12345);
    const a = estimateLag(flips, samples);
    const b = estimateLagEdges(flips, samples);
    expect(Math.abs(a.lagRefinedMs - b.plateau.centerMs)).toBeLessThan(6);
  });
});
