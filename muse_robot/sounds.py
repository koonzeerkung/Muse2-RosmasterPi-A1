"""Short beeps: optional feedback in `run --sound` (a tick when a command is detected).

Windows uses the built-in `winsound` (asynchronous, never blocks the EEG loop).
Elsewhere it tries afplay/aplay/paplay, and falls back to the terminal bell.
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

# name -> list of (freq Hz, duration s, pause after s)
TONES = {
    "event": [(1500, 0.06, 0)],
    "done": [(660, 0.1, 0.03), (880, 0.1, 0.03), (1100, 0.18, 0)],
}


def _make(name: str) -> Path:
    path = _DIR / f"{name}.wav"
    if path.exists():
        return path
    _DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for f, dur, pause in TONES[name]:
        t = np.arange(int(dur * RATE)) / RATE
        env = np.minimum(1, np.minimum(t, dur - t) / 0.01)            # 10 ms fade in/out
        s = 0.5 * np.sin(2 * np.pi * f * t) * env
        parts += [np.stack([s, s], axis=1), np.zeros((int(pause * RATE), 2))]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.concatenate(parts) * 32767).astype("<i2").tobytes())
    return path


class Sound:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.player = None
        if enabled and sys.platform != "win32":
            self.player = next((p for p in ("afplay", "aplay", "paplay") if shutil.which(p)), None)
        if enabled:
            try:
                for n in TONES:                                         # pre-generate: no delay later
                    _make(n)
            except OSError:
                self.enabled = False

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        try:
            path = _make(name)
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            elif self.player:
                args = [self.player, "-q", str(path)] if self.player == "aplay" else [self.player, str(path)]
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print("\a", end="", flush=True)
        except Exception:                                               # never let audio break the run
            pass
