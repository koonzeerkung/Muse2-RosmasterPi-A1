"""Audio cues for calibration (and optional feedback in `run`).

Why sound: during LOOK LEFT / LOOK RIGHT the eyes are off-screen, so a visual
"come back now" cue can't be seen. A sound from the speakers is heard wherever the
eyes point, and the "back" beep makes every trial's hold time the same, which keeps
the calibration data consistent. Nothing here blocks the EEG acquisition loop.

If a `signal.wav` is in the working folder (the project folder when started from
MuseRobot.bat), it is used as the "go" cue for look trials. Leading/trailing
silence is trimmed so the cue is instant and its length is exact. The "back" cue
stays a distinct low beep. (go_left / go_right panned variants are also built but
not used by calibration.)

Windows uses the built-in `winsound` (asynchronous). Elsewhere it tries
aplay/afplay/paplay, and falls back to the terminal bell.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 44100
_DIR = Path(tempfile.gettempdir()) / "muse_robot_sounds"

# name -> list of (freq Hz, duration s, left gain, right gain, pause after s)
TONES = {
    "go":       [(1000, 0.15, 1, 1, 0)],
    "go_left":  [(1000, 0.18, 1, 0, 0)],          # left ear only
    "go_right": [(1000, 0.18, 0, 1, 0)],          # right ear only
    "back":     [(600, 0.18, 1, 1, 0)],           # lower tone: back to center / release
    "block":    [(750, 0.08, 1, 1, 0.07), (750, 0.08, 1, 1, 0)],
    "done":     [(660, 0.1, 1, 1, 0.03), (880, 0.1, 1, 1, 0.03), (1100, 0.18, 1, 1, 0)],
    "event":    [(1500, 0.06, 1, 1, 0)],
}


def _write(path: Path, data: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(data, -32768, 32767).astype("<i2").tobytes())


def _custom_cues(path: Path) -> dict[str, Path]:
    """go / go_left / go_right made from the user's own 16-bit WAV file."""
    with wave.open(str(path)) as w:
        ch, rate, width, n = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("only 16-bit WAV supported")
    d = np.frombuffer(raw, "<i2").reshape(-1, ch).astype(float)
    d = np.repeat(d, 2, axis=1) if ch == 1 else d[:, :2]
    env = np.abs(d).max(axis=1)
    loud = np.where(env > 0.05 * env.max())[0]
    start = max(int(loud[0]) - int(0.005 * rate), 0)              # trim leading silence
    end = min(int(loud[-1]) + int(0.02 * rate), len(d))           # and trailing silence
    d = d[start:end]
    mono = d.mean(axis=1)
    tag = f"custom_{int(path.stat().st_mtime)}_{path.stat().st_size}"
    out = {}
    for name, stereo in (("go", d), ("go_left", np.stack([mono, 0 * mono], 1)),
                         ("go_right", np.stack([0 * mono, mono], 1))):
        out[name] = _DIR / f"{tag}_{name}.wav"
        if not out[name].exists():
            _write(out[name], stereo, rate)
    return out


def _make(name: str) -> Path:
    path = _DIR / f"{name}.wav"
    if path.exists():
        return path
    _DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for f, dur, gl, gr, pause in TONES[name]:
        t = np.arange(int(dur * RATE)) / RATE
        env = np.minimum(1, np.minimum(t, dur - t) / 0.01)            # 10 ms fade in/out
        s = 0.5 * np.sin(2 * np.pi * f * t) * env
        parts.append(np.stack([s * gl, s * gr], axis=1))
        parts.append(np.zeros((int(pause * RATE), 2)))
    _write(path, np.concatenate(parts) * 32767, RATE)
    return path


class Sound:
    def __init__(self, enabled: bool = True, custom: str | Path | None = "signal.wav"):
        self.enabled = enabled
        self.player = None
        self.files: dict[str, Path] = {}
        if enabled and sys.platform != "win32":
            self.player = next((p for p in ("afplay", "aplay", "paplay") if shutil.which(p)), None)
        if enabled:
            for n in TONES:                                             # pre-generate: no delay at cue time
                try:
                    self.files[n] = _make(n)
                except OSError:
                    self.enabled = False
            if custom and Path(custom).is_file():
                try:
                    self.files.update(_custom_cues(Path(custom)))
                    print(f"[sound] using {Path(custom).name} as the 'go' cue")
                except Exception as e:
                    print(f"[sound] could not use {custom}: {e} - using built-in beeps")

    def duration(self, name: str) -> float:
        """Length of the sound in seconds (0.2 s nominal when sound is off)."""
        if not self.enabled:
            return 0.2
        try:
            with wave.open(str(self.files.get(name) or _make(name))) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return 0.2

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        try:
            path = self.files.get(name) or _make(name)
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            elif self.player:
                args = [self.player, "-q", str(path)] if self.player == "aplay" else [self.player, str(path)]
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print("\a", end="", flush=True)
        except Exception:                                               # never let audio break recording
            pass
