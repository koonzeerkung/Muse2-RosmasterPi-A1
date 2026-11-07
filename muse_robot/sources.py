"""Where samples come from: Muse 2 via muselsl (LSL), via BrainFlow, or a simulator.

Every source returns float arrays of shape (n_samples, 4) in uV, channel order
TP9, AF7, AF8, TP10. `read()` may return an empty array when nothing is new.
If head-motion data is available (gyroscope), a 5th column holds the head
rotation speed resampled to the EEG samples (used to ignore head movements).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import signal

from .config import CHANNELS, FS

EMPTY = np.zeros((0, 4))


class Source:
    fs = FS

    def read(self) -> np.ndarray:
        raise NotImplementedError

    has_motion = False

    def cue(self, label: str, **kw) -> None:
        """Called by calibration when a cue is shown (used by the simulator)."""

    def flush(self) -> None:
        """Drop buffered samples, e.g. after waiting at a prompt, so cues stay in sync."""

    def close(self) -> None:
        pass


def _reorder(names: list[str]) -> list[int]:
    upper = [n.strip().upper() for n in names]
    try:
        return [upper.index(c) for c in CHANNELS]
    except ValueError:
        print(f"[source] channel labels {names} not recognised; using the first 4 as {CHANNELS}")
        return [0, 1, 2, 3]


class LSLSource(Source):
    """Muse 2 streamed by `muselsl stream` (or BlueMuse on Windows)."""

    def __init__(self, stream_type: str = "EEG", timeout: float = 15.0):
        from pylsl import StreamInlet, resolve_byprop

        print(f"[lsl] looking for an LSL stream of type '{stream_type}' ...")
        streams = resolve_byprop("type", stream_type, timeout=timeout)
        if not streams:
            raise RuntimeError("No EEG stream found. Start BlueMuse and press 'Start Streaming' "
                               "(or run `muselsl stream`).")
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        info = self.inlet.info()
        self.fs = info.nominal_srate() or FS
        if abs(self.fs - FS) > 1:
            print(f"[lsl] warning: stream is {self.fs} Hz, pipeline expects {FS} Hz")
        names, ch = [], info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        self.names = names
        self.idx = _reorder(names)
        print(f"[lsl] connected to '{info.name()}' channels={names}")

        # optional gyroscope stream (BlueMuse / muselsl can stream it next to EEG)
        self.gyro = None
        try:
            from pylsl import resolve_streams
            for s in resolve_streams(wait_time=1.0):
                if "gyro" in (s.name() + " " + s.type()).lower():
                    self.gyro = StreamInlet(s)
                    print(f"[lsl] head-motion stream: '{s.name()}' (head movements will be ignored)")
                    break
        except Exception as e:                                   # pragma: no cover
            print(f"[lsl] gyroscope lookup failed: {e}")
        if self.gyro is None:
            print("[lsl] no gyroscope stream (optional). Enable Gyroscope streaming in BlueMuse "
                  "to let the pipeline ignore head movements.")
        self.has_motion = self.gyro is not None
        self.g_ts = np.zeros(0)
        self.g_sp = np.zeros(0)

    def read_all(self) -> tuple[np.ndarray, np.ndarray]:
        """All channels (incl. AUX) + LSL timestamps, for diagnostics."""
        chunk, ts = self.inlet.pull_chunk(timeout=0.05, max_samples=512)
        if not chunk:
            return np.zeros((0, len(self.names))), np.zeros(0)
        return np.asarray(chunk, dtype=float), np.asarray(ts)

    def read(self) -> np.ndarray:
        chunk, ts = self.inlet.pull_chunk(timeout=0.05, max_samples=512)
        if not chunk:
            return EMPTY if self.gyro is None else np.zeros((0, 5))
        x = np.asarray(chunk, dtype=float)[:, self.idx]
        if self.gyro is None:
            return x
        g, gts = self.gyro.pull_chunk(timeout=0.0, max_samples=2048)
        if g:
            sp = np.linalg.norm(np.nan_to_num(np.asarray(g, dtype=float)[:, :3]), axis=1)
            self.g_ts = np.concatenate([self.g_ts, np.asarray(gts)])
            self.g_sp = np.concatenate([self.g_sp, sp])
            keep = self.g_ts > self.g_ts[-1] - 10.0
            self.g_ts, self.g_sp = self.g_ts[keep], self.g_sp[keep]
        motion = np.interp(np.asarray(ts), self.g_ts, self.g_sp) if len(self.g_ts) else np.zeros(len(x))
        return np.column_stack([x, motion])

    def flush(self) -> None:
        while self.inlet.pull_chunk(timeout=0.0, max_samples=4096)[0]:
            pass
        if self.gyro is not None:
            while self.gyro.pull_chunk(timeout=0.0, max_samples=4096)[0]:
                pass

    def close(self) -> None:
        self.inlet.close_stream()
        if self.gyro is not None:
            self.gyro.close_stream()


class BrainFlowSource(Source):
    """Muse 2 through BrainFlow's native Bluetooth support (no muselsl needed)."""

    def __init__(self, serial_number: str = "", dongle_port: str = ""):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        params = BrainFlowInputParams()
        params.serial_number = serial_number          # e.g. "Muse-1A2B" (optional)
        if dongle_port:                               # BLED112 dongle
            params.serial_port = dongle_port
            board_id = BoardIds.MUSE_2_BLED_BOARD.value
        else:
            board_id = BoardIds.MUSE_2_BOARD.value
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        self.rows = BoardShim.get_eeg_channels(board_id)
        names = BoardShim.get_eeg_names(board_id)
        self.rows = [self.rows[i] for i in _reorder(names)]
        self.fs = BoardShim.get_sampling_rate(board_id)
        print(f"[brainflow] streaming Muse 2, channels={names}")

    def read(self) -> np.ndarray:
        data = self.board.get_board_data()
        if data.shape[1] == 0:
            time.sleep(0.02)
            return EMPTY
        return data[self.rows].T.astype(float)

    def flush(self) -> None:
        self.board.get_board_data()

    def close(self) -> None:
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()


class SimulatedSource(Source):
    """Synthetic Muse 2 signal with realistic-ish artifacts.

    Background EEG + 50 Hz hum + sweat drift + natural blinks (~15/min).
    `inject(label)` / `cue(label)` add a gesture: JAW_CLENCH, DOUBLE_BLINK or EYEBROW_RAISE.
    `realtime=False` produces data as fast as possible.
    `motion=True` adds random head movements (EEG artifacts on all channels)
    and a 5th gyroscope-speed column, like LSLSource with a gyro stream.
    """

    def __init__(self, realtime: bool = True, seed: int = 0, chunk: int = 16,
                 natural_blinks: bool = True, reaction=(0.3, 0.6), motion: bool = False,
                 motion_every: float = 15.0):
        self.fs = FS
        self.has_motion = motion
        self.motion_every = motion_every
        self.rng = np.random.default_rng(seed)
        self.head_moves: list[float] = []
        self.realtime, self.chunk, self.reaction = realtime, chunk, reaction
        self.n = 0
        self.t0 = time.monotonic()
        self.active: list[list] = []          # [start_sample, waveform (m,4)]
        self.injected: list[tuple[str, float]] = []
        self.brown = np.zeros(4)
        self.offset = np.array([820.0, 790.0, 805.0, 830.0])   # raw DC offset like a real headset
        self.alpha_phase = self.rng.uniform(0, 2 * np.pi, 4)
        self.natural_blinks = natural_blinks
        self.next_nat = self._next_natural_gap()
        self.next_move = int(self.rng.exponential(motion_every) * FS) if motion else None
        self.emg_sos = signal.butter(4, 25, "highpass", fs=FS, output="sos")

    def _head_move(self) -> np.ndarray:
        """0.4-0.9 s head movement: electrode-shift artifact on every channel + gyro speed."""
        r = self.rng
        m = int(r.uniform(0.4, 0.9) * FS)
        shape = 0.5 * (1 - np.cos(2 * np.pi * np.arange(m) / m))       # smooth bump
        wf = np.zeros((m, 5))
        wf[:, :4] = shape[:, None] * r.normal(0, 60, 4)
        wf[:, 4] = shape * r.uniform(30, 80)                           # deg/s
        return wf

    # -- gesture waveforms (channels TP9, AF7, AF8, TP10) --------------------
    def _blink(self, amp: float) -> np.ndarray:
        # Measured on a real Muse 2: with the Fpz reference right above the eyes, a blink is a
        # large NEGATIVE dip on TP9/TP10 and only a small wiggle on AF7/AF8.
        t = np.arange(int(0.4 * FS)) / FS
        bump = amp * np.exp(-0.5 * ((t - 0.2) / 0.05) ** 2)
        return bump[:, None] * np.array([-1.0, 0.05, -0.2, -1.1])

    def _waveform(self, label: str) -> np.ndarray:
        r = self.rng
        if label == "DOUBLE_BLINK":
            gap = r.uniform(0.25, 0.45)
            a, b = self._blink(r.normal(170, 25)), self._blink(r.normal(170, 25))
            wf = np.zeros((len(a) + int(gap * FS), 4))
            wf[: len(a)] += a
            wf[int(gap * FS):] += b
            return wf
        if label == "EYEBROW_RAISE":
            # forehead muscle under AF7/AF8 (jaw muscles are under TP9/TP10) + slow skin shift
            dur = r.uniform(0.7, 1.2)
            m = int(dur * FS)
            noise = signal.sosfilt(self.emg_sos, r.normal(0, 1, (m + 200, 4)), axis=0)[200:]
            noise /= noise.std(axis=0)
            ramp = np.minimum(1, np.minimum(np.arange(m), np.arange(m)[::-1]) / (0.05 * FS))
            wf = noise * ramp[:, None] * np.array([18.0, 45.0, 45.0, 18.0])
            # measured on a real Muse 2: the forehead skin moves -> a slow shift of ~-700 uV on
            # AF7 (less on AF8) building up over ~0.25 s, held while the eyebrows are up
            t = np.arange(m) / FS
            shift = -r.uniform(500, 900) * np.clip((t - 0.08) / 0.2, 0, 1) * ramp
            wf += shift[:, None] * np.array([0.1, 1.0, 0.6, 0.1])
            return wf
        if label == "JAW_CLENCH":
            dur = r.uniform(0.7, 1.2)
            m = int(dur * FS)
            noise = signal.sosfilt(self.emg_sos, r.normal(0, 1, (m + 200, 4)), axis=0)[200:]
            noise /= noise.std(axis=0)
            ramp = np.minimum(1, np.minimum(np.arange(m), np.arange(m)[::-1]) / (0.05 * FS))
            wf = noise * ramp[:, None] * np.array([45.0, 15.0, 15.0, 45.0])
            t = np.arange(m) / FS     # low-frequency movement artifact at onset (asymmetric!)
            wf += np.exp(-0.5 * ((t - 0.1) / 0.06) ** 2)[:, None] * np.array([10.0, 25.0, 15.0, 10.0])
            return wf
        raise ValueError(label)

    def _next_natural_gap(self) -> int:
        return int((1.5 + self.rng.exponential(2.5)) * FS)

    # -- public ---------------------------------------------------------------
    def inject(self, label: str, delay: float = 0.0) -> None:
        start = self.n + int(delay * FS)
        self.active.append([start, self._waveform(label)])
        self.injected.append((label, start / FS))

    def cue(self, label: str, **kw) -> None:
        if label in ("JAW_CLENCH", "DOUBLE_BLINK", "EYEBROW_RAISE"):
            self.inject(label, delay=self.rng.uniform(*self.reaction))

    def flush(self) -> None:
        self.t0 = time.monotonic() - self.n / FS        # no backlog after waiting at a prompt

    def read(self) -> np.ndarray:
        ncol = 5 if self.has_motion else 4
        if self.realtime:
            k = min(int((time.monotonic() - self.t0) * FS) - self.n, 512)
            if k <= 0:
                time.sleep(0.01)
                return np.zeros((0, ncol))
        else:
            k = self.chunk
        r = self.rng
        x = np.zeros((k, 5))
        for i in range(k):                                   # brown-ish background noise
            self.brown = 0.98 * self.brown + r.normal(0, 2.5, 4)
            x[i, :4] = self.brown
        t = (self.n + np.arange(k)) / FS
        x[:, :4] += r.normal(0, 3, (k, 4))
        x[:, :4] += 5 * np.sin(2 * np.pi * 10 * t[:, None] + self.alpha_phase) * np.array([1, 0.3, 0.3, 1])
        x[:, :4] += 4 * np.sin(2 * np.pi * 50 * t)[:, None]
        x[:, :4] += 30 * np.sin(2 * np.pi * 0.05 * t)[:, None] * np.array([0.2, 1.0, 0.7, 0.2])
        x[:, :4] += self.offset
        x[:, 4] = np.abs(r.normal(0, 1.5, k))                # gyro noise when still

        if self.natural_blinks:
            while self.next_nat < self.n + k:
                self.active.append([self.next_nat, self._blink(r.normal(110, 20))])
                self.next_nat += self._next_natural_gap()
        if self.next_move is not None:
            while self.next_move < self.n + k:
                self.active.append([self.next_move, self._head_move()])
                self.head_moves.append(self.next_move / FS)
                self.next_move += int((2 + r.exponential(self.motion_every)) * FS)

        keep = []
        for start, wf in self.active:
            a, b = max(start, self.n), min(start + len(wf), self.n + k)
            if a < b:
                x[a - self.n: b - self.n, : wf.shape[1]] += wf[a - start: b - start]
            if start + len(wf) > self.n + k:
                keep.append([start, wf])
        self.active = keep
        self.n += k
        return x[:, :ncol]


def make_source(kind: str, fast: bool = False, seed: int = 0, **kw) -> Source:
    if kind == "lsl":
        return LSLSource()
    if kind == "brainflow":
        return BrainFlowSource(serial_number=kw.get("serial_number", ""), dongle_port=kw.get("dongle_port", ""))
    if kind == "sim":
        return SimulatedSource(realtime=not fast, seed=seed, motion=kw.get("motion", True))
    raise ValueError(f"unknown source '{kind}' (use lsl, brainflow or sim)")
