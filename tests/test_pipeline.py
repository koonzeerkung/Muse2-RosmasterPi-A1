"""Run with:  python -m pytest tests  (or just: python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse_robot.config import Profile  # noqa: E402
from muse_robot.detectors import CommandEngine  # noqa: E402
from muse_robot.sources import SimulatedSource  # noqa: E402


def test_chunking_is_identical():
    src = SimulatedSource(realtime=False, seed=3)
    raw = np.concatenate([src.read() for _ in range(400)])
    a, fa = CommandEngine(Profile()).process(raw)
    eng = CommandEngine(Profile())
    fb = [eng.process(raw[i:i + 12])[1].hd for i in range(0, len(raw), 12)]
    assert np.allclose(fa.hd, np.concatenate(fb))


def test_nan_does_not_poison_filters():
    src = SimulatedSource(realtime=False, seed=4)
    raw = np.concatenate([src.read() for _ in range(200)])
    raw[1000:1040, 1] = np.nan
    _, f = CommandEngine(Profile()).process(raw)
    assert np.isfinite(f.hd).all() and np.isfinite(f.emg).all()


def test_sim_end_to_end(tmp_path):
    from muse_robot import cli
    import argparse
    a = argparse.Namespace(profile=str(tmp_path / "p.json"), recordings=str(tmp_path), trials=8,
                           test_gestures=60, seed=0, no_gyro=False)
    acc, fp_per_min = cli.cmd_sim_test(a)       # includes head movements every ~15 s
    assert acc > 0.8 and fp_per_min < 0.5


def test_calibration_schedule():
    from muse_robot.calibrate import build_schedule
    msgs, markers, total = build_schedule(n_trials=8)
    assert total < 170                                           # about 2.5 min
    cues = [m for m in msgs if m["kind"] == "cue_on"]
    assert len(cues) == 32 and cues[0]["trial"] == 1 and cues[7]["n"] == 8
    backs = [m for m in msgs if m["kind"] == "cue_back"]
    assert len(backs) == 24                                      # jaw + left + right


def test_head_motion_blocks_look():
    from muse_robot.sources import SimulatedSource
    src = SimulatedSource(realtime=False, seed=9, natural_blinks=False, motion=False)
    eng = CommandEngine(Profile(gaze_thr=50, blink_thr=100, motion_thr=20))
    events = []
    for i in range(200):
        x = src.read()
        x = np.column_stack([x, np.full(len(x), 50.0 if 100 <= i < 140 else 1.0)])   # head turning
        if i == 105:
            src.inject("LOOK_LEFT")
        if i == 160:
            src.inject("LOOK_LEFT")
        events += eng.process(x)[0]
    looks = [e for e in events if e.gesture == "LOOK_LEFT"]
    assert len(looks) == 1 and looks[0].t > 160 * 16 / 256


def test_stop_survives_dropout_mute():
    """A jaw clench right after a Bluetooth gap must still give STOP."""
    from muse_robot.sources import SimulatedSource
    src = SimulatedSource(realtime=False, seed=5, natural_blinks=False)
    eng = CommandEngine(Profile(jaw_thr=20))
    events = []
    for i in range(160):
        x = src.read()
        if 100 <= i < 103:
            x[:] = np.nan                     # ~190 ms gap -> mute
        if i == 103:
            src.inject("JAW_CLENCH")
        events += eng.process(x)[0]
    assert any(e.gesture == "JAW_CLENCH" and 103 * 16 / 256 <= e.t <= 104 * 16 / 256 + 0.6 for e in events)


def test_robot_bridge_core():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot"))
    from eeg_cmd_bridge import BridgeCore
    c = BridgeCore(speed=0.15, steer=0.6, turn_time=1.5, link_timeout=1.5)
    c.on_command("GO", 0.0)
    assert c.velocity(0.1) == (0.15, 0.0)
    c.on_command("LEFT", 0.2)
    assert c.velocity(0.3) == (0.15, 0.6)
    c.on_packet(1.0)
    assert c.velocity(1.8) == (0.15, 0.0)          # turn finished, still driving
    c.on_command("STOP", 1.9)
    assert c.velocity(2.0) == (0.0, 0.0)
    c.on_command("GO", 2.1)
    assert c.velocity(3.7) == (0.0, 0.0)           # laptop silent > 1.5 s -> watchdog stop


if __name__ == "__main__":
    import tempfile
    test_chunking_is_identical()
    test_nan_does_not_poison_filters()
    test_calibration_schedule()
    test_head_motion_blocks_look()
    test_stop_survives_dropout_mute()
    test_robot_bridge_core()
    with tempfile.TemporaryDirectory() as d:
        test_sim_end_to_end(Path(d))
    print("all tests passed")
