"""Constants and the per-user calibration profile."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

FS = 256                                   # Muse 2 EEG sample rate (Hz)
CHANNELS = ("TP9", "AF7", "AF8", "TP10")   # order used everywhere in this package
CH = {name: i for i, name in enumerate(CHANNELS)}

GESTURES = ("JAW_CLENCH", "DOUBLE_BLINK", "EYEBROW_RAISE")

# Gesture -> command sent to the robot side (the packet format is in protocol.py)
DEFAULT_COMMAND_MAP = {
    "JAW_CLENCH": "TOGGLE_RUN",       # robot run / stop
    "DOUBLE_BLINK": "TOGGLE_MODE",    # gaze controls the camera / the robot movement
    "EYEBROW_RAISE": "TOGGLE_MUSIC",  # music on / off
}


@dataclass
class Profile:
    """Detector parameters. Defaults are rough; run `calibrate` for real use."""

    # Muscle burst (jaw clench or eyebrow raise): RMS of 30-110 Hz muscle activity,
    # averaged over all 4 channels (uV), above jaw_thr for >= jaw_min_dur
    jaw_thr: float = 20.0
    jaw_min_dur: float = 0.12      # s the envelope must stay above threshold
    jaw_refractory: float = 0.8    # s between two muscle events

    # Jaw vs eyebrow: raising the eyebrows moves the forehead skin under AF7/AF8, a large slow
    # shift of the forehead signal (measured: 640-960 uV within 0.3 s; jaw clench: < 45 uV).
    # brow_decide_s after the burst starts: shift clearly above brow_shift_thr (in direction
    # brow_dir) = EYEBROW_RAISE, clearly below = JAW_CLENCH. Within +-brow_margin of the
    # threshold the burst is unclear and IGNORED (both gestures are toggles, so a wrong
    # guess would flip the wrong thing).
    # 1e9 = not calibrated yet -> every burst is a jaw clench, reported without the wait.
    brow_shift_thr: float = 1e9    # uV
    brow_dir: int = -1             # -1: the forehead signal goes down when raising the eyebrows
    brow_decide_s: float = 0.3     # s after the burst starts (adds this delay to jaw clench when on)
    brow_margin: float = 0.25      # unclear zone: threshold x (1 - margin) .. x (1 + margin)

    # Blink signal B: 0.5-10 Hz EEG projected on the learned 4-channel blink pattern
    # (TP9, AF7, AF8, TP10). On Muse 2 the reference (Fpz) sits right above the eyes, so a
    # blink is mostly a large negative dip on TP9/TP10. Calibration re-learns it per user.
    blink_w: list = field(default_factory=lambda: [-0.9, 0.05, -0.2, -1.0])
    blink_thr: float = 100.0       # peak height of B (uV)
    blink_min_gap: float = 0.12    # s between the two blink peaks
    blink_max_gap: float = 0.6

    # Head motion (only if a gyroscope stream is available): rotation speed above this
    # blocks blink detections (moving the headset can look like a blink); muscle events are not blocked
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
