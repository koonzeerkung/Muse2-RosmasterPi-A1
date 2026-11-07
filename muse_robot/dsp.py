"""Causal, chunk-by-chunk signal processing.

Every filter keeps its state between calls, so feeding a recording in one big
chunk or in 12-sample chunks gives identical results (offline replay == live).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import CH, FS


class StreamFilter:
    """Second-order-sections IIR filter with persistent state over axis 0."""

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        if self.zi is None:
            # start in steady state for the first sample -> no start-up spike from DC offset
            zi0 = signal.sosfilt_zi(self.sos)                    # (sections, 2)
            shape = (zi0.shape[0], 2) + x.shape[1:]
            self.zi = zi0.reshape(zi0.shape + (1,) * (x.ndim - 1)) * x[0]
            self.zi = np.broadcast_to(self.zi, shape).copy()
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def notch_sos(fs: float, line: float) -> np.ndarray:
    parts = []
    for f0 in (line, 2 * line):
        if f0 < fs / 2 - 1:
            b, a = signal.iirnotch(f0, Q=30.0, fs=fs)
            parts.append(signal.tf2sos(b, a))
    return np.vstack(parts)


@dataclass
class Features:
    emg: np.ndarray    # (n,)  muscle (30-110 Hz) RMS envelope, mean of the 4 channels, uV
    emg_ear: np.ndarray   # (n,) same, ear sensors TP9/TP10 only (jaw muscles)
    emg_fore: np.ndarray  # (n,) same, forehead sensors AF7/AF8 only (eyebrow muscle)
    v: np.ndarray      # (n,)  blink signal B: 0.5-10 Hz EEG projected on the learned blink pattern, uV
    v_up: np.ndarray   # (n,)  B minus its 250 ms trailing minimum (blink height above local baseline)
    eeg: np.ndarray    # (n,4) 1-40 Hz EEG (for plotting / signal quality / future focus-relax)
    bad: np.ndarray    # (n,)  True where the raw sample had NaN/inf (filled with last good value)
    eye: np.ndarray | None = None  # (n,4) 0.5-10 Hz EEG used for the blink signal (calibration)
    motion: np.ndarray | None = None  # (n,) head-rotation speed (deg/s), zeros if no gyroscope


SIDEWAYS_EYE = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2)   # AF7 - AF8: sideways eye movement


def blink_axis(blink_w) -> np.ndarray:
    """Unit projection vector (4 channels) for the blink signal.

    blink_w is the 4-channel pattern of a blink learned in calibration. On Muse 2 the
    reference (Fpz) is right above the eyes, so a blink appears mostly as a large
    negative deflection on TP9/TP10 and only weakly on AF7/AF8. The sideways-eye
    direction (AF7 - AF8) is removed, so glancing left/right does not look like a
    blink (on the real recordings: fewer false double blinks, better cross-session).
    """
    w = np.asarray(blink_w, float)
    w = w / np.linalg.norm(w)
    w = w - (w @ SIDEWAYS_EYE) * SIDEWAYS_EYE
    return w / np.linalg.norm(w)


class FeatureExtractor:
    def __init__(self, fs: float = FS, line_freq: float = 50.0, blink_w=(-0.9, 0.05, -0.2, -1.0)):
        self.fs = fs
        self.b_axis = blink_axis(blink_w)
        self.notch = StreamFilter(notch_sos(fs, line_freq))
        self.emg_bp = StreamFilter(signal.butter(4, [30, min(110, fs / 2 - 8)], "bandpass", fs=fs, output="sos"))
        self.eye_bp = StreamFilter(signal.butter(2, [0.5, 10], "bandpass", fs=fs, output="sos"))
        self.eeg_bp = StreamFilter(signal.butter(4, [1, 40], "bandpass", fs=fs, output="sos"))

        self.env_n = int(0.10 * fs)                      # 100 ms RMS window
        self.env_zi = np.zeros((self.env_n - 1, 4))
        self.base_n = int(0.25 * fs)                     # blink baseline window
        self.v_hist = np.zeros(self.base_n - 1)
        self.last_good = None

    def _clean(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace NaN/inf (dropped Bluetooth packets, BlueMuse gaps) with the last good value.

        IIR filters turn a single NaN into NaN forever, so this must run first.
        Returns cleaned x and a per-sample mask of samples that had any bad channel.
        """
        bad = ~np.isfinite(x)
        bad_any = bad.any(axis=1)
        if bad_any.any():
            x = x.copy()
            for c in range(x.shape[1]):
                col, b = x[:, c], bad[:, c]
                if not b.any():
                    continue
                idx = np.where(~b, np.arange(len(col)), -1)
                np.maximum.accumulate(idx, out=idx)          # index of last good sample
                fill = np.where(idx >= 0, col[np.maximum(idx, 0)], self.last_good[c])
                col[b] = fill[b]
        good_rows = np.where(~bad_any)[0]
        if good_rows.size:
            self.last_good = x[good_rows[-1]].copy()
        return x, bad_any

    def process(self, x: np.ndarray) -> Features:
        x = np.asarray(x, dtype=float)
        if self.last_good is None:
            finite = x[np.isfinite(x).all(axis=1)]
            self.last_good = finite[0].copy() if len(finite) else np.zeros(x.shape[1])
        x, bad = self._clean(x)
        xn = self.notch(x)

        emg = self.emg_bp(xn)
        power, self.env_zi = signal.lfilter(np.ones(self.env_n) / self.env_n, [1.0], emg ** 2, axis=0,
                                            zi=self.env_zi)       # per-channel 100 ms mean power
        power = np.maximum(power, 0.0)
        env = np.sqrt(power.mean(axis=1))
        env_ear = np.sqrt(power[:, [CH["TP9"], CH["TP10"]]].mean(axis=1))
        env_fore = np.sqrt(power[:, [CH["AF7"], CH["AF8"]]].mean(axis=1))

        eye = self.eye_bp(xn)                            # 0.5-10 Hz, all 4 channels
        v = eye @ self.b_axis                            # blink signal

        # blink height relative to the lowest point of the last 250 ms, so the
        # second blink of a quick pair is not shrunk by the filter rebound of the first
        vv = np.concatenate([self.v_hist, v])
        win = np.lib.stride_tricks.sliding_window_view(vv, self.base_n)
        v_up = v - win.min(axis=1)
        self.v_hist = vv[-(self.base_n - 1):]

        return Features(emg=env, emg_ear=env_ear, emg_fore=env_fore, v=v, v_up=v_up,
                        eeg=self.eeg_bp(xn), bad=bad, eye=eye)
