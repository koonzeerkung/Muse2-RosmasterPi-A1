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
    fb = [eng.process(raw[i:i + 12])[1].v for i in range(0, len(raw), 12)]
    assert np.allclose(fa.v, np.concatenate(fb))


def test_nan_does_not_poison_filters():
    src = SimulatedSource(realtime=False, seed=4)
    raw = np.concatenate([src.read() for _ in range(200)])
    raw[1000:1040, 1] = np.nan
    _, f = CommandEngine(Profile()).process(raw)
    assert np.isfinite(f.v).all() and np.isfinite(f.emg).all()


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
    assert total < 120                                           # about 1.9 min
    cues = [m for m in msgs if m["kind"] == "cue_on"]
    assert len(cues) == 24 and cues[0]["trial"] == 1 and cues[7]["n"] == 8
    assert {m.label for m in markers} == {"REST", "JAW_CLENCH", "DOUBLE_BLINK", "EYEBROW_RAISE"}


def test_head_motion_blocks_blink():
    """A double blink while the head is turning is ignored; the same blink later is not."""
    from muse_robot.sources import SimulatedSource
    src = SimulatedSource(realtime=False, seed=9, natural_blinks=False, motion=False)
    eng = CommandEngine(Profile(blink_thr=100, motion_thr=20))
    events = []
    for i in range(200):
        x = src.read()
        x = np.column_stack([x, np.full(len(x), 50.0 if 100 <= i < 140 else 1.0)])   # head turning
        if i in (110, 160):
            src.inject("DOUBLE_BLINK")
        events += eng.process(x)[0]
    blinks = [e for e in events if e.gesture == "DOUBLE_BLINK"]
    assert len(blinks) == 1 and blinks[0].t > 160 * 16 / 256


def test_uncalibrated_profile_never_gives_eyebrow():
    """Without eyebrow calibration every muscle burst is a jaw clench, even a forehead one."""
    from muse_robot.sources import SimulatedSource
    src = SimulatedSource(realtime=False, seed=6, natural_blinks=False)
    eng = CommandEngine(Profile(jaw_thr=15))
    events = []
    for i in range(120):
        if i in (10, 70):
            src.inject("EYEBROW_RAISE" if i == 10 else "JAW_CLENCH")
        events += eng.process(src.read())[0]
    assert [e.gesture for e in events] == ["JAW_CLENCH", "JAW_CLENCH"]


def test_eyebrow_vs_jaw_after_calibration():
    from muse_robot.sources import SimulatedSource
    src = SimulatedSource(realtime=False, seed=6, natural_blinks=False)
    eng = CommandEngine(Profile(jaw_thr=15, brow_shift_thr=200, brow_dir=-1))
    events = []
    for i in range(120):
        if i in (10, 70):
            src.inject("EYEBROW_RAISE" if i == 10 else "JAW_CLENCH")
        events += eng.process(src.read())[0]
    assert [e.gesture for e in events] == ["EYEBROW_RAISE", "JAW_CLENCH"]


def test_old_recording_with_look_trials_still_fits():
    """Recordings made before look detection was removed still calibrate."""
    from muse_robot.calibrate import Marker, evaluate, fit_profile
    from muse_robot.sources import SimulatedSource
    src = SimulatedSource(realtime=False, seed=2)
    raw = np.concatenate([src.read() for _ in range(16 * 50)])
    markers = [Marker("REST", 256, 15 * 256)] + [Marker("LOOK_LEFT", 20 * 256 + k * 512, 512) for k in range(5)]
    p = fit_profile(raw, markers)
    ev = evaluate(raw, markers, p)
    assert "LOOK_LEFT" not in ev["confusion"] and "false_events_in_other_trials" in ev


def test_stop_survives_dropout_mute():
    """A jaw clench right after a Bluetooth gap must still be reported."""
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


def test_packets_and_toggles():
    import json
    from muse_robot.protocol import PacketBuilder, ToggleState
    pb = PacketBuilder(ToggleState())
    hb = pb.heartbeat()
    assert hb["type"] == "heartbeat" and hb["seq"] == 1 and hb["state"] == {"run": False, "mode": "robot", "music": False}
    m = pb.event("JAW_CLENCH", 1.0)
    assert m["event"]["command"] == "TOGGLE_RUN" and m["state"]["run"] is True and m["seq"] == 2
    m = pb.event("DOUBLE_BLINK", 2.0)
    assert m["event"]["command"] == "TOGGLE_MODE" and m["state"]["mode"] == "camera"
    m = pb.event("EYEBROW_RAISE", 3.0)
    assert m["event"]["command"] == "TOGGLE_MUSIC" and m["state"] == {"run": True, "mode": "camera", "music": True}
    assert pb.event("UNCLEAR_MUSCLE", 4.0) is None and pb.seq == 4     # never sent
    json.dumps(m)                                                         # serialisable


def test_example_receiver_follows_state_and_link():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "receiver"))
    import example_receiver as er
    from muse_robot.protocol import PacketBuilder, ToggleState
    calls = []
    er.on_run = lambda v: calls.append(("run", v))
    er.on_mode = lambda v: calls.append(("mode", v))
    er.on_music = lambda v: calls.append(("music", v))
    rx = er.Receiver()
    pb = PacketBuilder(ToggleState())
    rx.handle(pb.heartbeat(), 0.0)
    rx.handle(pb.event("JAW_CLENCH", 1), 0.5)
    pb.event("EYEBROW_RAISE", 2)                     # this packet is lost ...
    rx.handle(pb.heartbeat(), 1.0)                   # ... the next heartbeat still turns music on
    assert calls == [("run", True), ("music", True)]
    rx.check_link(3.0)                               # 2 s of silence -> stop
    assert calls[-1] == ("run", False) and rx.state["run"] is False


def test_unclear_muscle_burst_is_not_a_command():
    from muse_robot.detectors import MuscleDetector
    d = MuscleDetector(thr=10, min_dur=0.05, refractory=0.5, brow_shift_thr=400, brow_dir=-1, margin=0.25)
    out = []
    for i in range(200):                              # burst with a 400 uV forehead shift = on the line
        t = i / 256
        out += d.update(t, 50.0, 1.0, 1.0, -400.0 * min(i / 40, 1))
    assert [e.gesture for e in out] == ["UNCLEAR_MUSCLE"]


if __name__ == "__main__":
    import tempfile
    test_chunking_is_identical()
    test_nan_does_not_poison_filters()
    test_calibration_schedule()
    test_head_motion_blocks_blink()
    test_uncalibrated_profile_never_gives_eyebrow()
    test_eyebrow_vs_jaw_after_calibration()
    test_old_recording_with_look_trials_still_fits()
    test_stop_survives_dropout_mute()
    test_packets_and_toggles()
    test_example_receiver_follows_state_and_link()
    test_unclear_muscle_burst_is_not_a_command()
    with tempfile.TemporaryDirectory() as d:
        test_sim_end_to_end(Path(d))
    print("all tests passed")
