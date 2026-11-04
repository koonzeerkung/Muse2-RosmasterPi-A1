"""Causal, chunk-by-chunk signal processing.

Every filter keeps its state between calls, so feeding a recording in one big
chunk or in 12-sample chunks gives identical results (offline replay == live).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import FS


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
    emg: np.ndarray    # (n,)  jaw-muscle RMS envelope, uV
    v: np.ndarray      # (n,)  blink signal B: 0.5-10 Hz EEG projected on the learned blink direction, uV
    v_up: np.ndarray   # (n,)  B minus its 250 ms trailing minimum (blink height above local baseline)
    v_down: np.ndarray # (n,)  250 ms trailing maximum minus B
    hd: np.ndarray     # (n,)  gaze signal H: projected on the learned left-right direction (+ = left), uV
    h: np.ndarray      # (n,)  same as hd (kept for plotting code)
    eeg: np.ndarray    # (n,4) 1-40 Hz EEG (for plotting / signal quality / future focus-relax)
    bad: np.ndarray    # (n,)  True where the raw sample had NaN/inf (filled with last good value)
    eye: np.ndarray | None = None  # (n,4) 0.5-10 Hz EEG used for the eye signals (calibration)
    motion: np.ndarray | None = None  # (n,) head-rotation speed (deg/s), zeros if no gyroscope


def eye_axes(blink_w, gaze_w) -> tuple[np.ndarray, np.ndarray]:
    """Unit projection vectors (4 channels) for blink and gaze, each blind to the other.

    blink_w / gaze_w are the 4-channel patterns learned in calibration. On Muse 2 the
    reference (Fpz) is right above the eyes, so a blink appears mostly as a large
    negative deflection on TP9/TP10, and left/right eye movement as opposite spikes
    on AF7/AF8. Removing the shared part means a blink does not move H and a look
    does not move B.
    """
    ub = np.asarray(blink_w, float)
    ug = np.asarray(gaze_w, float)
    ub, ug = ub / np.linalg.norm(ub), ug / np.linalg.norm(ug)
    b_axis = ub - (ub @ ug) * ug
    g_axis = ug - (ug @ ub) * ub
    return b_axis / np.linalg.norm(b_axis), g_axis / np.linalg.norm(g_axis)


class FeatureExtractor:
    def __init__(self, fs: float = FS, line_freq: float = 50.0,
                 blink_w=(-0.9, 0.05, -0.2, -1.0), gaze_w=(0.2, 1.0, -1.0, -0.2)):
        self.fs = fs
        self.b_axis, self.g_axis = eye_axes(blink_w, gaze_w)
        self.notch = StreamFilter(notch_sos(fs, line_freq))
        self.emg_bp = StreamFilter(signal.butter(4, [30, min(110, fs / 2 - 8)], "bandpass", fs=fs, output="sos"))
        self.eye_bp = StreamFilter(signal.butter(2, [0.5, 10], "bandpass", fs=fs, output="sos"))
        self.eeg_bp = StreamFilter(signal.butter(4, [1, 40], "bandpass", fs=fs, output="sos"))

        self.env_n = int(0.10 * fs)                      # 100 ms RMS window
        self.env_zi = np.zeros(self.env_n - 1)
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
        power = np.mean(emg ** 2, axis=1)
        power, self.env_zi = signal.lfilter(np.ones(self.env_n) / self.env_n, [1.0], power, zi=self.env_zi)
        env = np.sqrt(np.maximum(power, 0.0))

        eye = self.eye_bp(xn)                            # 0.5-10 Hz, all 4 channels
        v = eye @ self.b_axis                            # blink signal
        h = eye @ self.g_axis                            # gaze signal (+ = left)
        hd = h

        # blink height relative to the lowest/highest point of the last 250 ms, so the
        # second blink of a quick pair is not shrunk by the filter rebound of the first
        vv = np.concatenate([self.v_hist, v])
        win = np.lib.stride_tricks.sliding_window_view(vv, self.base_n)
        v_up = v - win.min(axis=1)
        v_down = win.max(axis=1) - v
        self.v_hist = vv[-(self.base_n - 1):]

        return Features(emg=env, v=v, v_up=v_up, v_down=v_down, hd=hd, h=h, eeg=self.eeg_bp(xn), bad=bad, eye=eye)
