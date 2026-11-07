"""Real-time gesture detectors and the command engine that ties them together.

JAW_CLENCH /  muscle (30-110 Hz) envelope stays above threshold for >= jaw_min_dur;
EYEBROW_RAISE the burst is an eyebrow raise if the forehead signal moved more than
              brow_shift_thr within brow_decide_s, otherwise a jaw clench (= STOP)
DOUBLE_BLINK  two peaks of the blink signal B above threshold, blink_min_gap..blink_max_gap apart

B comes from dsp.FeatureExtractor (learned 4-channel blink pattern).
Cross-suppression: muscle bursts and head movement (gyroscope, if streamed) shift
the electrodes and create slow artifacts, so blink candidates that overlap them
are discarded. Muscle events (JAW_CLENCH = STOP) are never suppressed or muted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import CH, FS, Profile
from .dsp import FeatureExtractor, Features


@dataclass
class Event:
    gesture: str
    t: float                       # seconds since stream start (sample clock)
    info: dict = field(default_factory=dict)


class MuscleDetector:
    """Muscle bursts: JAW_CLENCH or EYEBROW_RAISE.

    A burst = muscle envelope above `thr` for `min_dur`. Raising the eyebrows moves the
    forehead skin under AF7/AF8, which shows as a large slow shift of the forehead signal
    (hundreds of uV within ~0.3 s); clenching the jaw barely moves it. So, when eyebrow
    detection is calibrated, the detector waits until `decide_s` after the burst started
    and calls it EYEBROW_RAISE only if the forehead moved more than `brow_shift_thr` in the
    learned direction `brow_dir`; everything else is JAW_CLENCH (= STOP, the safe side).
    Without eyebrow calibration every burst is JAW_CLENCH, reported right away.
    """

    BASE_N = 13         # ~50 ms of forehead signal before the burst = baseline

    def __init__(self, thr: float, min_dur: float, refractory: float,
                 brow_shift_thr: float | None = None, brow_dir: int = -1, decide_s: float = 0.3,
                 fs: float = FS):
        self.thr, self.min_n, self.refractory = thr, int(min_dur * fs), refractory
        self.brow_on = brow_shift_thr is not None and brow_shift_thr < 1e8
        self.brow_shift_thr, self.brow_dir, self.decide_s = brow_shift_thr, brow_dir, decide_s
        self.soft = 0.6 * thr
        self.count = 0
        self.ear_sum = self.fore_sum = 0.0
        self.hist: deque = deque(maxlen=self.BASE_N)
        self.run_onset = None
        self.run_base = 0.0
        self.run_exc = 0.0
        self.pending = None            # burst waiting for the jaw/eyebrow decision
        self.soft_start = None
        self.fired = False
        self.last_fire = -1e9
        self.intervals: deque = deque(maxlen=20)   # (start, end) of bursts that fired

    def update(self, t: float, env: float, ear: float = 1.0, fore: float = 0.0,
               fore_slow: float = 0.0) -> list[Event]:
        """env/ear/fore: muscle envelopes (all, TP9/TP10, AF7/AF8); fore_slow: 0.5-10 Hz forehead."""
        out = []
        if env > self.soft:
            if self.soft_start is None:
                self.soft_start = t
        else:
            if self.fired:
                self.intervals.append((self.soft_start, t))
            self.soft_start, self.fired = None, False
        if env > self.thr:
            if self.count == 0:                     # a new run above threshold starts here
                self.run_onset = t
                self.run_base = float(np.mean(self.hist)) if self.hist else fore_slow
                self.run_exc = 0.0
                self.ear_sum = self.fore_sum = 0.0
            self.count += 1
            self.ear_sum += ear
            self.fore_sum += fore
        else:
            self.count = 0
        if self.run_onset is not None:
            self.run_exc = max(self.run_exc, self.brow_dir * (fore_slow - self.run_base))
        if self.pending is not None:
            p = self.pending
            p["exc"] = max(p["exc"], self.brow_dir * (fore_slow - p["base"]))
            if t >= p["at"]:
                self.pending = None
                g = "EYEBROW_RAISE" if p["exc"] > self.brow_shift_thr else "JAW_CLENCH"
                out.append(Event(g, t, {"env": p["env"], "ratio": p["ratio"], "shift": round(p["exc"], 1)}))
        if self.count >= self.min_n and not self.fired and t - self.last_fire >= self.refractory:
            self.fired, self.last_fire = True, t
            ratio = round(float(np.log(max(self.fore_sum, 1e-6) / max(self.ear_sum, 1e-6))), 2)
            if not self.brow_on:
                out.append(Event("JAW_CLENCH", t, {"env": round(float(env), 1), "ratio": ratio}))
            else:
                self.pending = {"at": self.run_onset + self.decide_s, "base": self.run_base,
                                "exc": self.run_exc, "env": round(float(env), 1), "ratio": ratio}
        self.hist.append(fore_slow)
        return out

    def busy_between(self, a: float, b: float, pad: float = 0.3) -> bool:
        """Was there a (detected) muscle burst overlapping [a - pad, b + pad]?"""
        spans = list(self.intervals)
        if self.fired:
            spans.append((self.soft_start, float("inf")))
        return any(s <= b + pad and e >= a - pad for s, e in spans)


class MotionTracker:
    """Head-rotation speed above threshold -> 'moving' intervals (same API as MuscleDetector)."""

    def __init__(self, thr: float):
        self.thr = thr
        self.start = None
        self.intervals: deque = deque(maxlen=20)

    def update(self, t: float, speed: float) -> None:
        if speed > self.thr:
            if self.start is None:
                self.start = t
        elif self.start is not None:
            self.intervals.append((self.start, t))
            self.start = None

    def busy_between(self, a: float, b: float, pad: float = 0.3) -> bool:
        spans = list(self.intervals)
        if self.start is not None:
            spans.append((self.start, float("inf")))
        return any(s <= b + pad and e >= a - pad for s, e in spans)


def _blocked(blockers, a: float, b: float, pad: float = 0.3) -> bool:
    return any(x.busy_between(a, b, pad) for x in blockers)


class BlinkDetector:
    def __init__(self, p: Profile, blockers):
        self.thr = p.blink_thr
        self.min_gap, self.max_gap = p.blink_min_gap, p.blink_max_gap
        self.blockers = blockers
        self.in_peak = False
        self.start = self.peak_t = self.peak_v = 0.0
        self.pending = None
        self.refractory_until = -1e9

    def update(self, t: float, pv: float) -> list[Event]:
        """pv = blink height relative to local baseline (Features.v_up)."""
        out = []
        if not self.in_peak:
            if pv > self.thr:
                self.in_peak, self.start, self.peak_t, self.peak_v = True, t, t, pv
        else:
            if pv > self.peak_v:
                self.peak_t, self.peak_v = t, pv
            if pv < 0.5 * self.thr:
                self.in_peak = False
                if 0.04 <= t - self.start <= 0.6:
                    out += self._on_peak(self.peak_t, t)
        if self.pending is not None and t - self.pending > self.max_gap:
            self.pending = None
        return out

    def _on_peak(self, pt: float, now: float) -> list[Event]:
        if now < self.refractory_until:
            return []
        if _blocked(self.blockers, pt - 0.2, now, pad=0.5):
            self.pending = None
            return []
        if self.pending is not None and self.min_gap <= pt - self.pending <= self.max_gap:
            gap = pt - self.pending
            self.pending, self.refractory_until = None, now + 0.5
            return [Event("DOUBLE_BLINK", now, {"gap": round(gap, 3)})]
        self.pending = pt
        return []


class CommandEngine:
    """Raw Muse samples in -> gesture events out. Use one instance per stream.

    NaN handling: NaN samples are replaced by the last good value (in dsp). Short
    gaps are harmless; after a gap >= GAP_S the signal jumps when it comes back,
    which could look like a blink, so blink events are muted for MUTE_S.
    Muscle events (JAW_CLENCH = STOP, EYEBROW_RAISE) always get through.
    """

    GAP_S = 0.10
    MUTE_S = 0.6

    def __init__(self, profile: Profile, fs: float = FS):
        self.p, self.fs = profile, fs
        self.fx = FeatureExtractor(fs, profile.line_freq, profile.blink_w)
        self.jaw = MuscleDetector(profile.jaw_thr, profile.jaw_min_dur, profile.jaw_refractory,
                                  profile.brow_shift_thr, profile.brow_dir, profile.brow_decide_s, fs)
        self.motion = MotionTracker(profile.motion_thr)
        self.blink = BlinkDetector(profile, [self.jaw, self.motion])
        self.n = 0
        self.mute_until = 0.0          # no events while the signal recovers from a dropout
        self.bad_run = 0
        self.nan_samples = 0           # total samples that contained NaN (for status / UI)

    def process(self, x: np.ndarray) -> tuple[list[Event], Features]:
        """x: (n, 4) EEG, or (n, 5) with head-rotation speed (deg/s) in column 5."""
        x = np.asarray(x, dtype=float)
        motion = np.nan_to_num(x[:, 4]) if x.shape[1] >= 5 else None
        f = self.fx.process(x[:, :4])
        f.motion = motion if motion is not None else np.zeros(len(x))
        bh = f.v_up
        fore_slow = (f.eye[:, CH["AF7"]] + f.eye[:, CH["AF8"]]) / 2
        events: list[Event] = []
        self.nan_samples += int(f.bad.sum())
        gap_n = int(self.GAP_S * self.fs)
        for i in range(len(f.v)):
            t = (self.n + i) / self.fs
            if f.bad[i]:
                self.bad_run += 1
                if self.bad_run >= gap_n:
                    self.mute_until = t + self.MUTE_S
            else:
                self.bad_run = 0
            if motion is not None:
                self.motion.update(t, motion[i])
            new = self.jaw.update(t, f.emg[i], f.emg_ear[i], f.emg_fore[i], fore_slow[i])
            new += self.blink.update(t, bh[i])
            events += [e for e in new if t >= self.mute_until or e.gesture != "DOUBLE_BLINK"]
        self.n += len(f.v)
        return events, f


JawDetector = MuscleDetector     # old name
