/*
 * eye_timing_rig — photodiode edge timestamper + LED relay for validating the
 * browser eye-tracking timing correction.
 *
 * Board: Waveshare RP2040-Zero (any RP2040 board; pins are RP2040 GPIO numbers).
 * Core:  arduino-pico (earlephilhower), USB stack "Pico SDK".
 *
 * Hardware
 *   GP26 (ADC0)  TEMT6000 SIG (VCC->3V3, GND->GND), taped flat on the screen, shaded.
 *   GP15         LED anode via 120 ohm; cathode -> GND. Red or green (not white: a
 *                white LED's Vf leaves almost no current from 3.3 V). Clipped to the
 *                monitor bezel beside the webcam, in frame, pointed at the lens.
 *
 * What it does
 *   1. Samples the photodiode at 20 kHz from a hardware timer, tracks a slow
 *      min/max envelope, and timestamps every 50 % crossing (with hysteresis) with
 *      time_us_64(). These are the moments photons actually changed on the panel.
 *      Edge events and the relay alarm are raised inside the sampler IRQ, so a
 *      stalled main loop (e.g. serial back-pressure) can neither delay nor drop them.
 *      Serial writes are non-blocking; lines that do not fit are counted in
 *      `dropped_lines` (see `i`).
 *   2. Optionally relays each edge to the LED after a programmable delay D (fixed
 *      or random per edge), using absolute-time hardware alarms so main-loop
 *      latency never enters the measurement. The LED on/off instants are recorded.
 *   3. Optionally runs a free LED pulse train (random gaps), unsynchronised with
 *      anything, for mapping the camera's exposure window.
 *   4. Answers sync pings so the host can map rig microseconds onto its own clock.
 *
 * The browser drives it over Web Serial and logs every event line next to its own
 * rAF flip times and camera-frame captureTime values; the analysis is done there.
 *
 * Line protocol (ASCII, newline-terminated, all times in rig microseconds)
 *   -> rig                                    <- rig
 *   i                  info                   I {json}
 *   s <token>          sync ping              S <token> <t_us>
 *   m <0|1|2>          relay: off / rising edges / both edges
 *   d <us>             fixed relay delay
 *   r <min_us> <max_us> random relay delay per edge (uniform)
 *   w <us>             LED pulse width
 *   t <0|1>            free pulse train off / on
 *   g <min_ms> <max_ms> train gap range
 *   p                  pulse the LED now
 *   e <0|1>            edge reporting off / on
 *   a <us>             photodiode EMA time constant
 *
 *   Events
 *   E <t_us> <R|F> <level> <env_min> <env_max>      photodiode edge at the 50 % crossing
 *   L <t_on_us> <t_off_us> <src> <delay_us> <edge_t_us>
 *       src: R/F relayed from a rising/falling edge, T train, P manual.
 *       delay_us / edge_t_us are 0 for T and P.
 *   W <msg>                                         warning (e.g. no photodiode signal)
 */

#include "hardware/adc.h"
#include "hardware/gpio.h"
#include "pico/time.h"

// ------------------------------------------------------------------ config
static const uint PIN_PHOTO = 26;  // ADC0
static const uint PIN_LED = 15;

static const int32_t SAMPLE_PERIOD_US = 50;        // 20 kHz photodiode sampling
static const float ENVELOPE_TAU_S = 30.0f;          // min/max envelope leak (slow: a 1 s hold must not move the 50 % point)
static const int MIN_SWING_COUNTS = 80;             // below this, no signal -> no edges
static const float ARM_LO = 0.35f, MID = 0.50f, ARM_HI = 0.65f;

// ------------------------------------------------------------------ state
static volatile float ema = 0, ema_alpha = 0;       // photodiode EMA
static volatile float env_min = 4095, env_max = 0;
static float env_leak = 0;                          // per-sample leak toward current
static uint32_t ema_tau_us = 200;

enum PdState { PD_UNKNOWN, PD_LOW_ARMED, PD_LOW, PD_HIGH_ARMED, PD_HIGH };
static volatile PdState pd_state = PD_UNKNOWN;

// Boot defaults: relay every rising edge to the LED 100-140 ms later, with no serial
// involvement at all. The spread must cover at least one camera frame (33 ms): the
// display and camera run at an almost exact 2:1 ratio, so a fixed delay would land at
// the same camera phase every time and bias the mean residual by up to half a frame.
// The host may change these before a run; it never needs to.
static volatile int relay_mode = 1;                 // 0 off, 1 rising, 2 both
static volatile uint32_t delay_min_us = 100000, delay_max_us = 140000;
static uint32_t pulse_us = 100000;
static bool train_on = false;
static uint32_t gap_min_ms = 600, gap_max_ms = 1200;
static uint64_t train_next_us = 0;
static bool report_edges = true;
static uint32_t dropped_lines = 0;                  // serial lines discarded because the host was not draining
static uint32_t rng = 0x2545F491u;

static volatile bool led_busy = false;

// ------------------------------------------------------------------ event queue
enum EvType : uint8_t { EV_EDGE, EV_LED, EV_WARN };
struct Event {
  EvType type;
  char src;            // edge: 'R'/'F'; led: 'R','F','T','P'
  uint64_t t0;         // edge time, or LED on time
  uint64_t t1;         // LED off time
  uint32_t delay_us;
  uint64_t edge_t;     // LED: the edge it was relayed from
  int level, emin, emax;
};
static const int QN = 64;
static Event q[QN];
static volatile int q_head = 0, q_tail = 0;

static void q_push(const Event& e) {
  int next = (q_head + 1) % QN;
  if (next == q_tail) return;  // drop on overflow; host will notice the gap
  q[q_head] = e;
  q_head = next;
}

static uint32_t next_rand() {
  rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
  return rng;
}

static uint32_t rand_between(uint32_t lo, uint32_t hi) {
  return hi > lo ? lo + next_rand() % (hi - lo + 1) : lo;
}

// ------------------------------------------------------------------ LED pulses
struct Pulse {
  char src;
  uint32_t delay_us;
  uint64_t edge_t;
  uint64_t t_on;
};
static Pulse pulse_slot;   // one pulse in flight at a time

static int64_t led_off_cb(alarm_id_t, void*) {
  gpio_put(PIN_LED, 0);
  uint64_t t_off = time_us_64();
  Event e{};
  e.type = EV_LED;
  e.src = pulse_slot.src;
  e.t0 = pulse_slot.t_on;
  e.t1 = t_off;
  e.delay_us = pulse_slot.delay_us;
  e.edge_t = pulse_slot.edge_t;
  q_push(e);
  led_busy = false;
  return 0;
}

static int64_t led_on_cb(alarm_id_t, void*) {
  gpio_put(PIN_LED, 1);
  pulse_slot.t_on = time_us_64();
  add_alarm_in_us(pulse_us, led_off_cb, NULL, true);
  return 0;
}

/** Schedule one LED pulse at absolute rig time `at_us` (past times fire immediately). */
static bool schedule_pulse(uint64_t at_us, char src, uint32_t delay_us, uint64_t edge_t) {
  if (led_busy) return false;
  led_busy = true;
  pulse_slot.src = src;
  pulse_slot.delay_us = delay_us;
  pulse_slot.edge_t = edge_t;
  add_alarm_at(from_us_since_boot(at_us), led_on_cb, NULL, true);
  return true;
}

// ------------------------------------------------------------------ photodiode sampler (IRQ)
/**
 * Edge handling runs here, in the sampler IRQ: the event is queued and the relay alarm is
 * scheduled at an absolute time from the edge timestamp. Nothing the main loop does (in
 * particular, a serial write stalled by a host that is not draining) can delay or drop it.
 */
static void on_edge(char dir) {
  uint64_t t = time_us_64();
  Event e{};
  e.type = EV_EDGE;
  e.src = dir;
  e.t0 = t;
  e.level = (int)ema;
  e.emin = (int)env_min;
  e.emax = (int)env_max;
  q_push(e);
  bool relay = relay_mode == 2 || (relay_mode == 1 && dir == 'R');
  if (relay) {
    uint32_t d = rand_between(delay_min_us, delay_max_us);
    schedule_pulse(t + d, dir, d, t);
  }
}

static bool sample_cb(repeating_timer_t*) {
  uint16_t raw = adc_read();
  float v = ema + ema_alpha * ((float)raw - ema);
  ema = v;

  // Slow envelope: snap outward instantly, leak inward slowly.
  if (v > env_max) env_max = v; else env_max -= env_leak * (env_max - v);
  if (v < env_min) env_min = v; else env_min -= env_leak * (env_min - v);

  float swing = env_max - env_min;
  if (swing < MIN_SWING_COUNTS) { pd_state = PD_UNKNOWN; return true; }
  float lo = env_min + ARM_LO * swing;
  float mid = env_min + MID * swing;
  float hi = env_min + ARM_HI * swing;

  switch (pd_state) {
    case PD_UNKNOWN:
      pd_state = v > mid ? PD_HIGH : PD_LOW;
      break;
    case PD_LOW:
      if (v < lo) pd_state = PD_LOW_ARMED;
      break;
    case PD_LOW_ARMED:
      if (v > mid) {
        pd_state = PD_HIGH;
        on_edge('R');
      }
      break;
    case PD_HIGH:
      if (v > hi) pd_state = PD_HIGH_ARMED;
      break;
    case PD_HIGH_ARMED:
      if (v < mid) {
        pd_state = PD_LOW;
        on_edge('F');
      }
      break;
  }
  return true;
}

static void set_ema_tau(uint32_t tau_us) {
  ema_tau_us = tau_us < (uint32_t)SAMPLE_PERIOD_US ? SAMPLE_PERIOD_US : tau_us;
  ema_alpha = (float)SAMPLE_PERIOD_US / (float)ema_tau_us;
}

// ------------------------------------------------------------------ serial
static void print_info() {
  Serial.printf(
      "I {\"device\":\"eye_timing_rig\",\"version\":1,\"sample_hz\":%d,"
      "\"relay_mode\":%d,\"delay_min_us\":%lu,\"delay_max_us\":%lu,\"pulse_us\":%lu,"
      "\"train\":%d,\"gap_min_ms\":%lu,\"gap_max_ms\":%lu,\"report_edges\":%d,"
      "\"ema_tau_us\":%lu,\"env_min\":%d,\"env_max\":%d,\"pd_level\":%d,\"dropped_lines\":%lu,\"t_us\":%llu}\n",
      1000000 / SAMPLE_PERIOD_US, relay_mode, (unsigned long)delay_min_us,
      (unsigned long)delay_max_us, (unsigned long)pulse_us, train_on ? 1 : 0,
      (unsigned long)gap_min_ms, (unsigned long)gap_max_ms, report_edges ? 1 : 0,
      (unsigned long)ema_tau_us, (int)env_min, (int)env_max, (int)ema,
      (unsigned long)dropped_lines, (unsigned long long)time_us_64());
}

static char line[64];
static int line_len = 0;

static void handle_line(char* s) {
  char cmd = s[0];
  char* args = s + 1;
  unsigned long a = 0, b = 0;
  int n = sscanf(args, "%lu %lu", &a, &b);
  switch (cmd) {
    case 'i': print_info(); break;
    case 's': Serial.printf("S %lu %llu\n", a, (unsigned long long)time_us_64()); break;
    case 'm': relay_mode = (int)constrain((long)a, 0L, 2L); break;
    case 'd': delay_min_us = delay_max_us = a; break;
    case 'r': if (n == 2 && b >= a) { delay_min_us = a; delay_max_us = b; } break;
    case 'w': if (a >= 100) pulse_us = a; break;
    case 't': train_on = a != 0; train_next_us = 0; break;
    case 'g': if (n == 2 && b >= a && a > 0) { gap_min_ms = a; gap_max_ms = b; } break;
    case 'p': schedule_pulse(time_us_64(), 'P', 0, 0); break;
    case 'e': report_edges = a != 0; break;
    case 'a': set_ema_tau(a); break;
    default: break;
  }
}

static void poll_serial() {
  while (Serial.available()) {
    int c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (line_len > 0) { line[line_len] = 0; handle_line(line); }
      line_len = 0;
    } else if (line_len < (int)sizeof(line) - 1) {
      line[line_len++] = (char)c;
    }
  }
}

/** Write a line only if the USB CDC buffer can take all of it; otherwise count it as dropped. */
static void emit(const char* line) {
  size_t n = strlen(line);
  if ((size_t)Serial.availableForWrite() >= n) Serial.write((const uint8_t*)line, n);
  else dropped_lines++;
}

static void drain_events() {
  char line[96];
  while (q_tail != q_head) {
    Event e = q[q_tail];
    q_tail = (q_tail + 1) % QN;
    if (e.type == EV_EDGE) {
      if (!report_edges) continue;
      snprintf(line, sizeof line, "E %llu %c %d %d %d\n", (unsigned long long)e.t0, e.src,
               e.level, e.emin, e.emax);
      emit(line);
    } else if (e.type == EV_LED) {
      snprintf(line, sizeof line, "L %llu %llu %c %lu %llu\n", (unsigned long long)e.t0,
               (unsigned long long)e.t1, e.src, (unsigned long)e.delay_us,
               (unsigned long long)e.edge_t);
      emit(line);
    }
  }
}

// ------------------------------------------------------------------ main
static repeating_timer_t sampler;

void setup() {
  Serial.begin(115200);  // baud is ignored over USB CDC

  gpio_init(PIN_LED);
  gpio_set_dir(PIN_LED, GPIO_OUT);
  gpio_set_drive_strength(PIN_LED, GPIO_DRIVE_STRENGTH_12MA);
  gpio_put(PIN_LED, 0);

  adc_init();
  adc_gpio_init(PIN_PHOTO);
  adc_select_input(0);
  ema = adc_read();
  set_ema_tau(ema_tau_us);
  env_leak = (float)SAMPLE_PERIOD_US / (ENVELOPE_TAU_S * 1e6f);

  add_repeating_timer_us(-SAMPLE_PERIOD_US, sample_cb, NULL, &sampler);
}

void loop() {
  poll_serial();

  if (train_on) {
    uint64_t now = time_us_64();
    if (train_next_us == 0) train_next_us = now + 1000ull * rand_between(gap_min_ms, gap_max_ms);
    if (now >= train_next_us) {
      schedule_pulse(train_next_us, 'T', 0, 0);
      train_next_us = now + 1000ull * rand_between(gap_min_ms, gap_max_ms);
    }
  }

  drain_events();
}
