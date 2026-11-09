# Timing rig: photodiode + LED on an RP2040

Hardware ground truth for the browser eye-tracking timing correction. The browser's
loopback (`panel/` (see below)) measures display lag + camera lag as one number, anchored
on photons. This rig checks that number with an external clock and answers what the loopback
cannot: the display-side/camera-side split, the camera's exposure window, and whether the
LCD's slow falling edge biases the loopback's falling-edge estimates.

The board is a **timestamping relay**, not a sample streamer: it timestamps every photodiode
edge on its own microsecond clock, fires an LED into the webcam a programmable delay after
each edge (or on a free-running schedule), and reports every event as a text line. The
browser drives it over Web Serial and logs the events next to its own rAF flip times and
camera-frame `captureTime` values.

## The panel (`panel/`)

The browser half of the rig, and the home of the screen->webcam loopback it validates.
Standalone Vite app — it has no dependency on the model or on `@saccadejs/core`:

```
cd web/rig/panel && npm install
npm run dev        # open the printed URL, then "Connect rig" (Web Serial, Chrome only)
npm test           # 54 tests: loopback estimator + rig analysis
```

`src/loopback.ts` is a copy of the shipping estimator in `jspsych/saccadejs`
(`packages/core/src/loopback.ts`) — see the provenance header in that file. The rig is what
validated it, so `diff` the two before trusting a measurement if they have drifted.

## Hardware

Same parts as the `jspsych-audio-latency` rig (RP2040-Zero + TEMT6000), plus one LED.

```
3V3 ── TEMT6000 VCC
GND ── TEMT6000 GND
GP26 ── TEMT6000 SIG              (ADC0; the breakout has its own 10 kΩ load)

GP15 ── 120 Ω ── LED anode
                 LED cathode ── GND
```

- **LED colour: red or green, not white.** A white LED's forward voltage (~3 V) leaves almost
  no current from a 3.3 V pin. Red at 120 Ω gives ~11 mA; the firmware raises the pin's drive
  strength to 12 mA (the default is 4 mA).
- **Placement.** Photodiode taped flat on the screen under a black hood (room light will
  otherwise swamp the swing). LED clipped to the monitor bezel beside the webcam, inside the
  camera's frame, off to one side of the face, pointed at the lens. A scrap of tape over it
  as a diffuser prevents saturation, which matters for the exposure-window measurement.
- **Display brightness at maximum.** Many panels PWM-dim below 100 %; the firmware's EMA
  (200 µs default) will not remove a few-hundred-hertz flicker and edges will be garbage.
  If the `E` lines look wrong, check the raw level with `i` while the screen is static.
- If you need a brighter LED (5 ms pulses are dim after a 33 ms exposure), drive it through
  an NPN transistor (1 kΩ base resistor from GP15) from 3V3 with a 47 Ω series resistor.

## Firmware

`firmware/eye_timing_rig/eye_timing_rig.ino`. Board core: arduino-pico (earlephilhower),
already installed as `rp2040:rp2040`.

```
cd web/rig/firmware
arduino-cli compile --fqbn rp2040:rp2040:waveshare_rp2040_zero:usbstack=picosdk eye_timing_rig
arduino-cli upload  --fqbn rp2040:rp2040:waveshare_rp2040_zero:usbstack=picosdk -p /dev/cu.usbmodem* eye_timing_rig
```

First flash: hold BOOT, tap RESET (or plug in while holding BOOT) so the board mounts as
`RPI-RP2`; later uploads go straight over USB.

### What it does

1. Samples the photodiode at 20 kHz from a hardware timer, tracks a slow min/max envelope
   (4 s leak), and timestamps every crossing of the 50 % level with `time_us_64()`. Hysteresis
   arms at 35 % / 65 % so LCD ripple cannot double-trigger. No calibration step: the
   envelope self-adjusts as soon as the screen has shown both levels.
2. Optionally relays each edge to the LED after delay **D** (fixed, or uniform-random per
   edge), using an **absolute-time** hardware alarm from the edge timestamp, so main-loop
   latency never shifts the pulse. LED on and off instants are recorded.
3. Optionally runs a free pulse train (random 0.6–1.2 s gaps), unsynchronised with anything.
4. Answers sync pings so the browser can fit rig-µs → `performance.now()`.

### Protocol (ASCII lines, times in rig µs)

| send | meaning | reply |
|---|---|---|
| `i` | info | `I {json}` |
| `s <token>` | sync ping | `S <token> <t_us>` |
| `m 0/1/2` | relay off / rising edges / both edges | |
| `d <us>` | fixed relay delay | |
| `r <min_us> <max_us>` | random relay delay per edge | |
| `w <us>` | LED pulse width (default 100 000) | |
| `t 0/1` | free pulse train | |
| `g <min_ms> <max_ms>` | train gap range | |
| `p` | pulse now | |
| `e 0/1` | edge reporting | |
| `a <us>` | photodiode EMA time constant | |

Events: `E <t_us> R|F <level> <env_min> <env_max>` for each photodiode edge, and
`L <t_on_us> <t_off_us> <src> <delay_us> <edge_t_us>` for each LED pulse (`src` R/F relayed
from an edge, T train, P manual).

Quick check from a terminal before wiring anything into the browser:

```
python3 -m serial.tools.miniterm /dev/cu.usbmodem* 115200
i          → I {...}   (env_min/env_max should separate by hundreds of counts once the screen flashes)
p          → LED blinks; an L line follows with a ~100 000 µs on/off gap
```

## Measurements

All three use the sparse (no-flicker) loopback run in `timing.html` as the stimulus. The rig
section of that page connects, configures, and logs the rig events into the same
JSON export as the flips and camera samples.

### 1. Display lag, and the LCD rise/fall asymmetry

Sync pings (20 at start, 20 at end; RTT over USB is ~1 ms so each maps to ±0.5 ms) fit
rig-µs onto the browser clock. Then every rAF flip pairs with its `E` line:
display lag = `E.t` (mapped) − flip time. Report rising and falling separately. The
audio-latency rig measured this panel at 15 ms rise, ~130 ms fall; if the falling edge really
is that slow, the loopback's light→dark edges are smeared over four camera frames and the
sparse estimator should switch to rising edges only.

### 2. The relay test: does the correction close the loop?

Relay mode `m 1`, random delay `r 100000 140000` (the boot default). The spread must cover at least one camera frame: display and camera run at an almost exact 2:1 ratio, so a fixed delay would land at the same camera phase every time and bias the mean by up to half a frame. For each LED pulse the
browser finds the first camera frame in which the LED is visible and takes its `captureTime`
**c**. Apply the loopback correction **L** (the 82 ms) and compare with the rAF flip of the
edge that triggered it:

```
(c − L) − t_flip  ≈  D          (+ exposure-window quantisation)
```

Display lag and camera lag cancel by construction, and no cross-clock mapping is involved:
the rig reports D, the browser reports the left-hand side. If the correction is right the
residual is zero-mean with a spread set by the exposure window. This is the direct test of
the number applied to gaze data, with zero physiology in the loop.

### 3. Exposure window

Train mode `t 1`, pulse width 5 ms (`w 5000`), LED diffused so it does not saturate. Pulse
phase relative to the camera's frame clock is uniformly random, so plotting the LED's
brightness in the first frame it appears in (as a fraction of its two-frame total) against the
sync-mapped pulse time modulo the frame period draws the exposure window directly: its width
is the exposure time, its position tells whether `captureTime` sits at the start, centre, or
end of integration. That residual, up to half a frame, is the last unknown in the correction.

## Browser side

Built into `panel/` (see below) (Chrome, for Web Serial): the "hardware rig" panel connects
to the board, syncs 20 pings before the countdown and again after capture, applies the
relay/train settings for the run, and logs every rig line next to the flips and camera
frames. The camera sampler carries a per-frame LED detector (`peak` = mean of the 24
brightest pixels at 160×120). The report gains a rig section with the sync fit, display lag
by edge direction, relay residuals with a delay→residual table, the exposure window, and a
verdict; all of it lands in the JSON export under `rig`. Client: `web/rig/panel/src/rig.ts`;
analysis (DOM-free, tested): `web/rig/panel/src/rigAnalysis.ts`.
