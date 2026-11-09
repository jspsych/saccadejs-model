// Web Serial client for the timing rig (web/rig).
//
// The rig is an RP2040 that timestamps every 50 % luminance crossing of a photodiode taped
// to the screen and fires an LED into the webcam's field of view a programmable delay after
// each edge. It speaks newline-terminated ASCII over USB CDC; the protocol is the header
// comment of web/rig/firmware/eye_timing_rig/eye_timing_rig.ino:
//
//   ->  i | s <token> | m 0|1|2 | d <us> | r <min> <max> | w <us> | t 0|1 | g <min> <max>
//       | p | e 0|1 | a <us>
//   <-  I {json} | S <token> <t_us>
//       E <t_us> R|F <level> <env_min> <env_max>
//       L <t_on_us> <t_off_us> <src> <delay_us> <edge_t_us>      src: R F T P
//       W <msg>
//
// Browser-only (DOM + Web Serial, i.e. Chrome). The analysis lives in ./rigAnalysis, which
// is DOM-free and unit-tested; nothing here does arithmetic beyond parsing.

import type { SyncPair } from "./rigAnalysis";

// ---------------------------------------------------------------- Web Serial typings
// Web Serial is not in lib.dom, so declare the minimal shape we use and reach it through a
// cast. Everything is feature-detected: on a browser without it the page still runs.

interface SerialOpenOptions {
  baudRate: number;
  bufferSize?: number;
}

interface SerialPortLike {
  readable: ReadableStream<Uint8Array> | null;
  writable: WritableStream<Uint8Array> | null;
  open(options: SerialOpenOptions): Promise<void>;
  close(): Promise<void>;
  getInfo?(): { usbVendorId?: number; usbProductId?: number };
}

interface SerialLike {
  requestPort(options?: unknown): Promise<SerialPortLike>;
  getPorts(): Promise<SerialPortLike[]>;
}

function getSerial(): SerialLike | null {
  const nav = navigator as unknown as { serial?: SerialLike };
  return nav.serial ?? null;
}

// ---------------------------------------------------------------- events

export type RigEventKind = "E" | "L" | "S" | "I" | "W" | "?";

export interface RigInfo {
  device?: string;
  version?: number;
  sample_hz?: number;
  relay_mode?: number;
  delay_min_us?: number;
  delay_max_us?: number;
  pulse_us?: number;
  train?: number;
  gap_min_ms?: number;
  gap_max_ms?: number;
  report_edges?: number;
  ema_tau_us?: number;
  env_min?: number;
  env_max?: number;
  pd_level?: number;
  t_us?: number;
  [key: string]: unknown;
}

interface RigEventCommon {
  /** `performance.now()` when the line finished arriving. */
  tArrive: number;
  raw: string;
}

export interface RigEdgeEvent extends RigEventCommon {
  kind: "E";
  tUs: number;
  dir: "R" | "F";
  level: number;
  envMin: number;
  envMax: number;
}

export interface RigLedEvent extends RigEventCommon {
  kind: "L";
  tOnUs: number;
  tOffUs: number;
  src: "R" | "F" | "T" | "P";
  delayUs: number;
  edgeTUs: number;
}

export interface RigSyncEvent extends RigEventCommon {
  kind: "S";
  token: number;
  tUs: number;
}

export interface RigInfoEvent extends RigEventCommon {
  kind: "I";
  info: RigInfo;
}

export interface RigWarnEvent extends RigEventCommon {
  kind: "W";
  message: string;
}

export interface RigOtherEvent extends RigEventCommon {
  kind: "?";
}

export type RigEvent =
  RigEdgeEvent | RigLedEvent | RigSyncEvent | RigInfoEvent | RigWarnEvent | RigOtherEvent;

const SYNC_TIMEOUT_MS = 500;
const INFO_TIMEOUT_MS = 1000;
const SYNC_GAP_MIN_MS = 20;
const SYNC_GAP_MAX_MS = 40;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isLedSrc(s: string): s is "R" | "F" | "T" | "P" {
  return s === "R" || s === "F" || s === "T" || s === "P";
}

/** One rig line -> one event. Anything unrecognised comes back as kind "?" with its text. */
export function parseRigLine(raw: string, tArrive: number): RigEvent {
  const f = raw.trim().split(/\s+/);
  const kind = f[0];
  if (kind === "E" && f.length >= 6) {
    const dir = f[2] === "F" ? "F" : "R";
    return {
      kind: "E",
      tArrive,
      raw,
      tUs: Number(f[1]),
      dir,
      level: Number(f[3]),
      envMin: Number(f[4]),
      envMax: Number(f[5]),
    };
  }
  if (kind === "L" && f.length >= 6 && isLedSrc(f[3])) {
    return {
      kind: "L",
      tArrive,
      raw,
      tOnUs: Number(f[1]),
      tOffUs: Number(f[2]),
      src: f[3],
      delayUs: Number(f[4]),
      edgeTUs: Number(f[5]),
    };
  }
  if (kind === "S" && f.length >= 3) {
    return { kind: "S", tArrive, raw, token: Number(f[1]), tUs: Number(f[2]) };
  }
  if (kind === "I") {
    let info: RigInfo = {};
    const brace = raw.indexOf("{");
    if (brace >= 0) {
      try {
        info = JSON.parse(raw.slice(brace)) as RigInfo;
      } catch {
        info = { raw: raw.slice(brace) };
      }
    }
    return { kind: "I", tArrive, raw, info };
  }
  if (kind === "W") {
    return { kind: "W", tArrive, raw, message: raw.slice(1).trim() };
  }
  return { kind: "?", tArrive, raw };
}

// ---------------------------------------------------------------- client

export class Rig {
  /** Every line the rig has sent since the object was created, parsed. */
  readonly events: RigEvent[] = [];
  onEvent: ((e: RigEvent) => void) | null = null;
  lastInfo: RigInfo | null = null;

  private port: SerialPortLike | null = null;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
  private pump: Promise<void> | null = null;
  private readonly decoder = new TextDecoder();
  private readonly encoder = new TextEncoder();
  private buf = "";
  private nextToken = 1;
  private readonly waitingSync = new Map<number, (e: RigSyncEvent) => void>();
  private waitingInfo: ((e: RigInfoEvent) => void)[] = [];

  static isSupported(): boolean {
    return getSerial() !== null;
  }

  get connected(): boolean {
    return this.port !== null;
  }

  /** Something to show the user about which port this is; USB ids when the browser gives them. */
  portLabel(): string {
    const info = this.port?.getInfo?.();
    const vendor = info?.usbVendorId;
    const product = info?.usbProductId;
    if (vendor === undefined || product === undefined) return "serial port";
    const hex = (v: number): string => v.toString(16).padStart(4, "0");
    return `usb ${hex(vendor)}:${hex(product)}`;
  }

  /** Prompts for a port (needs a user gesture) and starts the line reader. */
  async connect(): Promise<void> {
    if (this.port) return;
    const serial = getSerial();
    if (!serial) throw new Error("Web Serial unavailable — use Chrome over https or localhost");
    const port = await serial.requestPort();
    // Chrome defaults to a 255-byte receive buffer; a busy page can let that back up to the
    // device. The firmware never blocks on it any more, but keep the pipe wide anyway.
    await port.open({ baudRate: 115200, bufferSize: 65536 });
    if (!port.readable || !port.writable) {
      await port.close();
      throw new Error("port opened without a readable/writable stream");
    }
    this.port = port;
    this.buf = "";
    this.writer = port.writable.getWriter();
    this.reader = port.readable.getReader();
    this.pump = this.readLoop(this.reader);
  }

  async disconnect(): Promise<void> {
    const reader = this.reader;
    this.reader = null;
    if (reader) {
      try {
        await reader.cancel();
      } catch {
        /* already gone */
      }
      try {
        reader.releaseLock();
      } catch {
        /* already released */
      }
    }
    if (this.pump) {
      try {
        await this.pump;
      } catch {
        /* the loop swallows its own errors */
      }
      this.pump = null;
    }
    const writer = this.writer;
    this.writer = null;
    if (writer) {
      try {
        writer.releaseLock();
      } catch {
        /* already released */
      }
    }
    const port = this.port;
    this.port = null;
    if (port) {
      try {
        await port.close();
      } catch {
        /* unplugged */
      }
    }
    this.waitingSync.clear();
    this.waitingInfo = [];
  }

  async send(line: string): Promise<void> {
    if (!this.writer) throw new Error("rig not connected");
    await this.writer.write(this.encoder.encode(`${line}\n`));
  }

  /** `i` — the firmware's state, including the photodiode envelope. */
  async info(): Promise<RigInfo> {
    const wait = new Promise<RigInfoEvent>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.waitingInfo = this.waitingInfo.filter((r) => r !== onInfo);
        reject(new Error("timed out waiting for I"));
      }, INFO_TIMEOUT_MS);
      const onInfo = (e: RigInfoEvent): void => {
        window.clearTimeout(timer);
        resolve(e);
      };
      this.waitingInfo.push(onInfo);
    });
    await this.send("i");
    const e = await wait;
    this.lastInfo = e.info;
    return e.info;
  }

  /**
   * `n` round-trip pings. Each pair brackets one rig timestamp between the write and the
   * reply, so `fitSync` can put the rig's microseconds on the browser clock. Sent one at a
   * time (a queued ping would inflate its own RTT) with a small random gap so the pings do
   * not lock onto anything periodic. A ping that times out is dropped, not retried.
   */
  async sync(n = 20): Promise<SyncPair[]> {
    const out: SyncPair[] = [];
    for (let i = 0; i < n; i++) {
      const token = this.nextToken++;
      const wait = new Promise<RigSyncEvent>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          this.waitingSync.delete(token);
          reject(new Error("sync ping timed out"));
        }, SYNC_TIMEOUT_MS);
        this.waitingSync.set(token, (e) => {
          window.clearTimeout(timer);
          resolve(e);
        });
      });
      const tSend = performance.now();
      try {
        await this.send(`s ${token}`);
        const e = await wait;
        out.push({ token, tSend, tRecv: e.tArrive, rigUs: e.tUs });
      } catch {
        /* dropped ping; the fit only needs most of them */
      }
      await sleep(SYNC_GAP_MIN_MS + Math.random() * (SYNC_GAP_MAX_MS - SYNC_GAP_MIN_MS));
    }
    return out;
  }

  /** 0 = off, 1 = relay rising edges to the LED, 2 = relay both. */
  setRelay(mode: 0 | 1 | 2): Promise<void> {
    return this.send(`m ${mode}`);
  }

  setDelay(fixedUs: number): Promise<void> {
    return this.send(`d ${uint(fixedUs)}`);
  }

  setDelayRange(minUs: number, maxUs: number): Promise<void> {
    const lo = uint(minUs);
    return this.send(`r ${lo} ${Math.max(lo, uint(maxUs))}`);
  }

  setPulseWidth(widthUs: number): Promise<void> {
    return this.send(`w ${Math.max(100, uint(widthUs))}`);
  }

  setTrain(on: boolean): Promise<void> {
    return this.send(`t ${on ? 1 : 0}`);
  }

  setGaps(minMs: number, maxMs: number): Promise<void> {
    const lo = Math.max(1, uint(minMs));
    return this.send(`g ${lo} ${Math.max(lo, uint(maxMs))}`);
  }

  pulse(): Promise<void> {
    return this.send("p");
  }

  setEdgeReporting(on: boolean): Promise<void> {
    return this.send(`e ${on ? 1 : 0}`);
  }

  // -------------------------------------------------------------- internals

  private async readLoop(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> {
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) this.feed(this.decoder.decode(value, { stream: true }));
      }
    } catch {
      /* cancelled, or the device went away; disconnect() cleans up */
    }
  }

  private feed(text: string): void {
    const tArrive = performance.now();
    this.buf += text;
    for (;;) {
      const nl = this.buf.indexOf("\n");
      if (nl < 0) break;
      const line = this.buf.slice(0, nl).replace(/\r$/, "");
      this.buf = this.buf.slice(nl + 1);
      if (line.length > 0) this.dispatch(parseRigLine(line, tArrive));
    }
    // A device that stops mid-line must not wedge the reader.
    if (this.buf.length > 4096) this.buf = "";
  }

  private dispatch(e: RigEvent): void {
    this.events.push(e);
    if (e.kind === "S") {
      const resolve = this.waitingSync.get(e.token);
      if (resolve) {
        this.waitingSync.delete(e.token);
        resolve(e);
      }
    } else if (e.kind === "I") {
      this.lastInfo = e.info;
      const waiting = this.waitingInfo;
      this.waitingInfo = [];
      for (const resolve of waiting) resolve(e);
    }
    this.onEvent?.(e);
  }
}

/** The firmware parses arguments with `%lu`: non-negative integers only. */
function uint(v: number): number {
  return Number.isFinite(v) ? Math.max(0, Math.round(v)) : 0;
}
