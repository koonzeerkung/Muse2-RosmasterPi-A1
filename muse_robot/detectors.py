"""Real-time gesture detectors and the command engine that ties them together.

JAW_CLENCH    muscle (30-110 Hz) envelope stays above threshold for >= jaw_min_dur
DOUBLE_BLINK  two peaks of the blink signal B above threshold, blink_min_gap..blink_max_gap apart
LOOK_LEFT/    a spike of the gaze signal H ("out"), then an opposite spike ("back")
LOOK_RIGHT    gaze_min_gap..gaze_max_gap later; the sign of the first spike is the direction

B and H come from dsp.FeatureExtractor (learned 4-channel blink / look patterns).
Cross-suppression: jaw clenching and head movement (gyroscope, if streamed) shift
the electrodes and create slow artifacts, so blink/look candidates that overlap
them are discarded. JAW_CLENCH (= STOP) is never suppressed or muted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import FS, Profile
from .dsp import FeatureExtractor, Features


@dataclass
class Event:
    gesture: str
    t: float                       # seconds since stream start (sample clock)
    info: dict = field(default_factory=dict)


class JawDetector:
    def __init__(self, thr: float, min_dur: float, refractory: float, fs: float = FS):
        self.thr, self.min_n, self.refractory = thr, int(min_dur * fs), refractory
        self.soft = 0.6 * thr
        self.count = 0
        self.soft_start = None
        self.fired = False
        self.last_fire = -1e9
        self.intervals: deque = deque(maxlen=20)   # (start, end) of bursts that fired

    def update(self, t: float, env: float) -> list[Event]:
        out = []
        if env > self.soft:
            if self.soft_start is None:
                self.soft_start = t
        else:
            if self.fired:
                self.intervals.append((self.soft_start, t))
            self.soft_start, self.fired, self.count = None, False, 0
        self.count = self.count + 1 if env > self.thr else 0
        if self.count >= self.min_n and not self.fired and t - self.last_fire >= self.refractory:
            self.fired, self.last_fire = True, t
            out.append(Event("JAW_CLENCH", t, {"env": round(float(env), 1)}))
        return out

    def busy_between(self, a: float, b: float, pad: float = 0.3) -> bool:
        """Was there a (detected) jaw clench overlapping [a - pad, b + pad]?"""
        spans = list(self.intervals)
        if self.fired:
            spans.append((self.soft_start, float("inf")))
        return any(s <= b + pad and e >= a - pad for s, e in spans)


class MotionTracker:
    """Head-rotation speed above threshold -> 'moving' intervals (same API as JawDetector)."""

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


class GazeDetector:
    QUIET_S = 1.0       # after a look, ignore spikes this long (leftovers of the return movement)

    def __init__(self, p: Profile, blockers):
        self.thr, self.sign = p.gaze_thr, p.gaze_left_sign
        self.min_gap, self.max_gap = p.gaze_min_gap, p.gaze_max_gap
        self.blockers = blockers
        self.quiet_until = -1e9        # ignore the rebound of the "back" spike after a look
        self.in_spike = False
        self.s = 0
        self.ext = self.ext_t = 0.0
        self.pending = None            # (sign, time)

    def update(self, t: float, hd: float) -> list[Event]:
        """hd = gaze signal H (+ = towards left)."""
        d = self.sign * hd
        out = []
        if not self.in_spike:
            if abs(d) > self.thr:
                self.in_spike, self.s, self.ext, self.ext_t = True, int(np.sign(d)), abs(d), t
        else:
            if self.s * d > self.ext:
                self.ext, self.ext_t = self.s * d, t
            if self.s * d < 0.5 * self.thr:
                self.in_spike = False
                out += self._on_spike(self.s, self.ext_t, self.ext, t)
        if self.pending is not None and t - self.pending[1] > self.max_gap:
            self.pending = None
        return out

    def _on_spike(self, s: int, ts: float, amp: float, now: float) -> list[Event]:
        # Blinks often come together with eye movements, so spikes near a blink are NOT
        # rejected. A double blink leaks into H as spikes ~0.3 s apart, which is shorter
        # than gaze_min_gap, so it cannot form a look.
        if ts < self.quiet_until:
            return []
        if self.pending is not None and self.pending[0] == -s and ts - self.pending[1] < self.min_gap:
            return []                         # filter rebound right after a spike, not a real move

        if _blocked(self.blockers, ts - 0.3, now):
            self.pending = None
            return []
        if self.pending is not None and self.pending[0] == -s and \
                self.min_gap <= ts - self.pending[1] <= self.max_gap:
            first = self.pending[0]
            if _blocked(self.blockers, self.pending[1] - 0.3, now):    # head moved during the look
                self.pending = None
                return []
            self.pending = None
            self.quiet_until = ts + self.QUIET_S
            return [Event("LOOK_LEFT" if first > 0 else "LOOK_RIGHT", now, {"amp": round(float(amp), 1)})]
        self.pending = (s, ts)
        return []


class CommandEngine:
    """Raw Muse samples in -> gesture events out. Use one instance per stream.

    NaN handling: NaN samples are replaced by the last good value (in dsp). Short
    gaps are harmless; after a gap >= GAP_S the signal jumps when it comes back,
    which could look like a blink/eye movement, so blink/look events are muted for
    MUTE_S. JAW_CLENCH (STOP) always gets through.
    """

    GAP_S = 0.10
    MUTE_S = 0.6

    def __init__(self, profile: Profile, fs: float = FS):
        self.p, self.fs = profile, fs
        self.fx = FeatureExtractor(fs, profile.line_freq, profile.blink_w, profile.gaze_w)
        self.jaw = JawDetector(profile.jaw_thr, profile.jaw_min_dur, profile.jaw_refractory, fs)
        self.motion = MotionTracker(profile.motion_thr)
        self.blink = BlinkDetector(profile, [self.jaw, self.motion])
        self.gaze = GazeDetector(profile, [self.jaw, self.motion])
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
            new = self.jaw.update(t, f.emg[i])
            new += self.blink.update(t, bh[i])
            new += self.gaze.update(t, f.hd[i])
            events += [e for e in new if t >= self.mute_until or e.gesture == "JAW_CLENCH"]
        self.n += len(f.v)
        return events, f
