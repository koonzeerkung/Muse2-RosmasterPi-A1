"""Constants and the per-user calibration profile."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

FS = 256                                   # Muse 2 EEG sample rate (Hz)
CHANNELS = ("TP9", "AF7", "AF8", "TP10")   # order used everywhere in this package
CH = {name: i for i, name in enumerate(CHANNELS)}

GESTURES = ("JAW_CLENCH", "DOUBLE_BLINK", "LOOK_LEFT", "LOOK_RIGHT")

# Gesture -> robot command. Edit to suit your robot.
DEFAULT_COMMAND_MAP = {
    "JAW_CLENCH": "STOP",
    "DOUBLE_BLINK": "GO",
    "LOOK_LEFT": "LEFT",
    "LOOK_RIGHT": "RIGHT",
}


@dataclass
class Profile:
    """Detector parameters. Defaults are rough; run `calibrate` for real use."""

    # Jaw clench: RMS of 30-110 Hz muscle activity, averaged over all 4 channels (uV)
    jaw_thr: float = 20.0
    jaw_min_dur: float = 0.12      # s the envelope must stay above threshold
    jaw_refractory: float = 0.8    # s between two jaw events

    # Eye signals: 0.5-10 Hz EEG projected on learned 4-channel patterns (TP9, AF7, AF8, TP10).
    # Defaults come from a real Muse 2 session; calibration re-learns them per user.
    blink_w: list = field(default_factory=lambda: [-0.9, 0.05, -0.2, -1.0])
    gaze_w: list = field(default_factory=lambda: [0.2, 1.0, -1.0, -0.2])     # pattern of a LEFT look

    # Blink: peaks of the blink signal B (uV)
    blink_thr: float = 100.0
    blink_polarity: int = 1        # kept for old profiles; the learned pattern makes blinks positive
    blink_min_gap: float = 0.12    # s between the two blink peaks
    blink_max_gap: float = 0.6

    # Gaze: spikes of the gaze signal H. On Muse 2 an eye movement gives a short spike
    # (+ = towards left), not a held step, so a look = "out" spike + opposite "back" spike.
    gaze_thr: float = 80.0
    blink_k: float = 1.0           # unused (old profiles)
    gaze_left_sign: int = 1        # +1: the learned pattern already makes LEFT positive
    gaze_min_gap: float = 0.45     # s between "look out" and "look back" spikes (> double-blink leak)
    gaze_max_gap: float = 2.5

    # Head motion (only if a gyroscope stream is available): rotation speed above this
    # blocks blink/look detections, so turning the head is not mistaken for an eye command
    motion_thr: float = 20.0       # deg/s

    line_freq: float = 50.0        # mains frequency (Thailand = 50 Hz)
    calibrated: bool = False
    report: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
