"""Command line entry point:  python -m muse_robot <command> ...

  check       stream 10 s and report sample rate, NaN, contact quality per channel
  calibrate   guided recording (~1.3 min) -> per-user profile JSON
  run         live detection -> commands (screen / Yahboom robot / UDP / serial)
  robot-test  drive the Yahboom robot from the keyboard (no headset)
  fit         re-fit a profile from a saved calibration recording
  sim-test    end-to-end test with the simulated headset (no hardware)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import numpy as np

from .calibrate import QuietUI, calibrate, evaluate, fit_profile, load_recording, print_report
from .config import CHANNELS, DEFAULT_COMMAND_MAP, FS, GESTURES, Profile
from .detectors import CommandEngine
from .outputs import make_sink
from .sounds import Sound
from .sources import LSLSource, SimulatedSource, Source, make_source


class _NoGyro(Source):
    """Simulator with head movements but without the gyroscope column (like BlueMuse with gyro off)."""

    def __init__(self, src):
        self.src = src

    def __getattr__(self, k):
        return getattr(self.src, k)

    def read(self):
        return self.src.read()[:, :4]

    def cue(self, label, **kw):
        self.src.cue(label, **kw)

    def flush(self):
        self.src.flush()


def _add_source_args(ap):
    ap.add_argument("--source", default="lsl", choices=["lsl", "brainflow", "sim"],
                    help="lsl = BlueMuse / muselsl stream (default), brainflow = direct Bluetooth, sim = simulator")
    ap.add_argument("--fast", action="store_true", help="simulator only: run faster than real time")
    ap.add_argument("--serial-number", default="", help="brainflow only: e.g. Muse-1A2B")
    ap.add_argument("--seed", type=int, default=0)


def cmd_check(a):
    if a.source != "lsl":
        print("check currently inspects LSL streams (BlueMuse / muselsl)")
        return
    src = LSLSource()
    data, ts = [], []
    t_end = time.monotonic() + a.seconds
    print(f"recording {a.seconds:.0f} s ...")
    while time.monotonic() < t_end:
        x, t = src.read_all()
        if len(x):
            data.append(x)
            ts.append(t)
    if not data or sum(len(d) for d in data) < FS:
        print("no EEG data arrived - is BlueMuse streaming (Start Streaming)?")
        return
    x, t = np.concatenate(data), np.concatenate(ts)
    dur = t[-1] - t[0]
    print(f"\nsamples={len(x)}  duration={dur:.1f}s  measured rate={len(x) / max(dur, 1e-6):.1f} Hz (expect {FS})")
    dt = np.diff(t)
    expected = dur * FS
    lost = max(expected - len(x), 0)
    print(f"largest timestamp gap = {dt.max() * 1000:.0f} ms (packets arriving in bursts is normal)")
    print(f"lost samples ~ {lost:.0f} of {expected:.0f} ({lost / expected * 100:.2f}%)"
          + ("   <-- Bluetooth dropouts: move closer to the PC" if lost / expected > 0.01 else "   OK"))

    from scipy import signal as sg
    from .dsp import notch_sos

    bp = sg.butter(4, [1, 40], "bandpass", fs=FS, output="sos")
    hum = sg.butter(4, [48, 52], "bandpass", fs=FS, output="sos")
    print(f"\n{'channel':8s} {'NaN %':>6s} {'max NaN run':>11s} {'drift p5-p95':>13s} "
          f"{'EEG 1-40Hz':>11s} {'50Hz hum':>9s} {'railed %':>9s}  verdict")
    for c, name in enumerate(src.names):
        if name.strip().upper() not in CHANNELS:
            continue
        col = x[:, c]
        nan = ~np.isfinite(col)
        run = longest = 0
        for b in nan:
            run = run + 1 if b else 0
            longest = max(longest, run)
        good = np.where(nan, np.nanmedian(col), col)
        drift = np.percentile(good, 95) - np.percentile(good, 5)
        eeg = sg.sosfiltfilt(bp, sg.sosfiltfilt(notch_sos(FS, 50.0), good - good.mean()))
        eeg_sd = np.median([np.std(w) for w in np.array_split(eeg, max(len(eeg) // FS, 1))])  # robust to blinks
        hum_rms = np.std(sg.sosfiltfilt(hum, good - good.mean()))
        lo, hi = good.min(), good.max()          # clipping: many samples stuck at the extremes
        railed = (np.mean(good <= lo + 0.5) + np.mean(good >= hi - 0.5)) * 100 if hi - lo > 1 else 100.0
        problems = []
        if railed > 1:
            problems.append("hitting the rail -> no contact")
        if hum_rms > 10:
            problems.append("strong 50 Hz -> poor contact")
        if eeg_sd > 50:
            problems.append("noisy")
        if drift > 400:
            problems.append("large drift -> sensor moving / drying")
        verdict = "OK" if not problems else ", ".join(problems)
        print(f"{name:8s} {nan.mean() * 100:6.2f} {longest:>7d} smp {drift:10.0f} uV "
              f"{eeg_sd:8.1f} uV {hum_rms:6.1f} uV {railed:8.1f}   {verdict}")
    print("\nEEG 1-40Hz = typical 1-s noise level (median, so blinks don't count). Normal: ~5-30 uV."
          "\nIf a channel is not OK: push the headband down so the forehead pads sit flat, move hair"
          "\naway from the ear sensors, wipe skin, or dampen the sensors slightly, then check again."
          "\nNaN in TP9/AF7/AF8/TP10: short runs are filled; runs >= 100 ms mute detection for 0.6 s.")


def cmd_calibrate(a):
    if a.trials < 5:
        print("using the minimum of 5 trials per gesture")
        a.trials = 5
    src = make_source(a.source, fast=a.fast, seed=a.seed, serial_number=a.serial_number)
    try:
        calibrate(src, a.profile, recording_dir=a.recordings, n_trials=a.trials, rest_s=a.rest,
                  sound=Sound(not a.no_sound))
    finally:
        src.close()


def cmd_fit(a):
    raw, markers = load_recording(a.recording)
    p = fit_profile(raw, markers)
    ev = evaluate(raw, markers, p)
    p.report["evaluation"], p.report["recording"] = ev, a.recording
    p.save(a.profile)
    print_report(p, ev)
    print(f"\nprofile saved to {a.profile}")


def cmd_run(a):
    try:
        profile = Profile.load(a.profile)
    except FileNotFoundError:
        print(f"profile {a.profile} not found - using rough defaults. Run `calibrate` first!")
        profile = Profile()
    src = make_source(a.source, fast=a.fast, seed=a.seed, serial_number=a.serial_number)
    sinks = [make_sink(s) for s in (a.output or ["print"])]
    engine = CommandEngine(profile)
    sound = Sound(a.sound)
    if src.has_motion:
        print(f"head-motion gating on (>{profile.motion_thr:.0f} deg/s blocks blink commands)")
    src.flush()
    qual = deque(maxlen=FS)
    last_status = last_hb = time.monotonic()
    robots = [s for s in sinks if hasattr(s, "heartbeat")]
    for r in robots:
        r.heartbeat()
    demo_next = 3.0
    rng = np.random.default_rng(a.seed)
    print("running - Ctrl+C to stop")
    try:
        while True:
            if robots and time.monotonic() - last_hb >= 0.5:      # keep the robot's watchdog happy
                last_hb = time.monotonic()
                for r in robots:
                    r.heartbeat()
            if isinstance(src, SimulatedSource) and src.n / FS >= demo_next:   # demo gestures
                src.inject(str(rng.choice(GESTURES)))
                demo_next += 4.0
            x = src.read()
            if not len(x):
                continue
            events, f = engine.process(x)
            qual.extend(f.eeg)
            for ev in events:
                sound.play("event")
                cmd = DEFAULT_COMMAND_MAP.get(ev.gesture, ev.gesture)
                for s in sinks:
                    s.send(ev, cmd)
            if time.monotonic() - last_status > 1.0 and len(qual) > 10:
                last_status = time.monotonic()
                sd = np.std(np.asarray(qual), axis=0)
                parts = [f"{n}:{v:5.1f}{'!' if v > 80 or v < 0.5 else ' '}" for n, v in zip(CHANNELS, sd)]
                robot = "  | " + " ".join(r.status_text() for r in robots) if robots else ""
                print("\r signal(uV) " + " ".join(parts) + f"  NaN={engine.nan_samples}{robot}    ",
                      end="", flush=True)
            if a.duration and engine.n / FS >= a.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        src.close()
        for s in sinks:
            s.close()


def cmd_robot_test(a):
    """Drive the robot from the keyboard (no headset) to check the Wi-Fi link and the bridge."""
    import threading

    from .outputs import UDPSink
    r = UDPSink(a.ip, a.port)
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            r.heartbeat()
            time.sleep(0.5)

    threading.Thread(target=beat, daemon=True).start()
    print(f"Robot test -> {a.ip}:{a.port}   (the robot must be running robot/start_bridge.sh)")
    time.sleep(1.0)
    print("  " + ("CONNECTED - " + r.status_text() if r.connected() else
                  "NO REPLY from the robot yet: check the IP, same Wi-Fi, and that the bridge is running"))
    keys = {"g": "GO", "w": "GO", "s": "STOP", " ": "STOP", "x": "STOP", "a": "LEFT", "l": "LEFT",
            "d": "RIGHT", "r": "RIGHT"}
    print("  type a key + Enter:  g=GO  s=STOP  a=LEFT  d=RIGHT  q=quit   (STOP is sent on quit)")
    try:
        while True:
            k = input("  > ").strip().lower()
            if k in ("q", "quit", "exit"):
                break
            cmd = keys.get(k[:1] if k else " ", k.upper())
            if cmd not in ("GO", "STOP", "LEFT", "RIGHT"):
                print("  ?  use g / s / a / d / q")
                continue
            r.send(None, cmd)
            time.sleep(0.25)
            r.poll()
            print("   ", r.status_text())
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        r.close()
        print("  sent STOP, bye")


def cmd_sim_test(a):
    """Calibrate on one simulated session, then test on a different one."""
    gyro = "with gyroscope" if not a.no_gyro else "WITHOUT gyroscope"
    print(f"1) simulated calibration session (natural blinks + random head movements, {gyro}) ...")
    src = SimulatedSource(realtime=False, seed=a.seed, motion=True)
    if a.no_gyro:
        src = _NoGyro(src)
    p = calibrate(src, a.profile, recording_dir=a.recordings, n_trials=a.trials,
                  ui=QuietUI(redo=True), sound=Sound(False))

    print("\n2) independent test session (random gestures every 4-6 s, natural blinks, "
          "head movements, NaN dropouts) ...")
    sim = SimulatedSource(realtime=False, seed=a.seed + 100, motion=True)
    src = _NoGyro(sim) if a.no_gyro else sim
    eng = CommandEngine(p)
    rng = np.random.default_rng(a.seed + 7)
    events, t_next, n_test = [], 3.0, a.test_gestures
    while sim.n < (3 + 5 * n_test + 3) * FS:
        if len(sim.injected) < n_test and sim.n >= t_next * FS:
            sim.inject(str(rng.choice(GESTURES)))
            t_next += rng.uniform(4, 6)
        x = src.read()
        if rng.random() < 0.01:                       # ~1 dropout every 6 s: 1-30 NaN samples
            k = int(rng.integers(1, 30))
            x[: min(k, len(x)), rng.integers(0, 4)] = np.nan
        e, _ = eng.process(x)
        events += e
    hit = {g: [0, 0] for g in GESTURES}
    matched = set()
    for lab, t in sim.injected:
        hit[lab][1] += 1
        cands = [i for i, ev in enumerate(events) if t <= ev.t <= t + 2.5 and i not in matched]
        if cands:
            matched.add(cands[0])
            hit[lab][0] += events[cands[0]].gesture == lab
    fp = len(events) - len(matched)
    total = sum(v[1] for v in hit.values())
    correct = sum(v[0] for v in hit.values())
    minutes = sim.n / FS / 60
    print(f"test accuracy {correct}/{total} = {correct / total * 100:.1f}%   " +
          ", ".join(f"{g}={v[0]}/{v[1]}" for g, v in hit.items()))
    print(f"unmatched (false) events: {fp} in {minutes:.1f} min   head movements: {len(sim.head_moves)}"
          f"   NaN samples: {eng.nan_samples}")
    return correct / total, fp / minutes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="muse_robot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="diagnose the LSL stream (NaN, rate, signal level)")
    _add_source_args(s)
    s.add_argument("--seconds", type=float, default=10)
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("calibrate", help="guided calibration -> profile")
    _add_source_args(s)
    s.add_argument("--profile", default="profiles/default.json")
    s.add_argument("--recordings", default="recordings")
    s.add_argument("--trials", type=int, default=8)
    s.add_argument("--rest", type=float, default=15)
    s.add_argument("--no-sound", action="store_true", help="no 'done' chime at the end")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("run", help="live detection -> robot")
    _add_source_args(s)
    s.add_argument("--profile", default="profiles/default.json")
    s.add_argument("--output", action="append",
                   help="print | robot:IP[:PORT] | udp:HOST:PORT | serial:PORT[:BAUD]  (repeatable)")
    s.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = forever)")
    s.add_argument("--sound", action="store_true", help="short beep whenever a command is detected")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("robot-test", help="drive the Yahboom robot from the keyboard (check the link)")
    s.add_argument("ip", help="robot IP address")
    s.add_argument("--port", type=int, default=9000)
    s.set_defaults(func=cmd_robot_test)

    s = sub.add_parser("fit", help="re-fit profile from a saved recording")
    s.add_argument("recording")
    s.add_argument("--profile", default="profiles/default.json")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("sim-test", help="end-to-end test with simulated headset")
    s.add_argument("--profile", default="profiles/sim.json")
    s.add_argument("--recordings", default="recordings")
    s.add_argument("--trials", type=int, default=8)
    s.add_argument("--test-gestures", type=int, default=80)
    s.add_argument("--no-gyro", action="store_true", help="simulate BlueMuse without the gyroscope stream")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sim_test)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
