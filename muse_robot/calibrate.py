"""Calibration: cue the user (screen + sound), record, fit per-user thresholds, evaluate.

Protocol (default 8 trials, about 2.6 min):
  15 s rest (eyes open, blink normally)
  4 blocks (jaw, double blink, look left, look right) of N trials
  jaw / blink trial = 2.0 s cue window + 1.5 s pause
  look trial        = go sound + 1 s silence + "eyes back" beep + 1.2 s, then 1.5 s pause

Cues
  jaw / double blink: screen only (">>> NOW <<<", "relax jaw"), no sound
  look left / right:  go sound (signal.wav if present, else a beep) from the speakers,
                      then, 1 s AFTER the go sound has finished, a low "eyes back" beep.
                      The direction comes from the block (all-left, then all-right).
  The eyes are off-screen during look trials, so the return cue has to be audible;
  the 1 s gap keeps the two sounds clearly separate.

Robustness
  * Natural blinks are expected everywhere: calibration learns the 4-channel pattern
    of a blink and of a look and makes each signal blind to the other; natural
    blinks in the rest/look data count as "not a double blink".
  * Trials spoiled by head movement / big electrode shifts / Bluetooth gaps are
    detected and left out of the threshold fitting (reported as "rejected").
  * Gestures that still come out weak can be re-recorded on their own (redo).

The raw recording + cue markers are saved as .npz so you can re-fit later or use
them to train an ML model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks

from .config import CH, CHANNELS, FS, GESTURES, Profile
from .detectors import CommandEngine
from .dsp import FeatureExtractor
from .sounds import Sound

LEAD_S = 3.0          # filters settle; no cues
REST_S = 15.0
BLOCK_PAUSE_S = 3.0   # read the instruction
CUE_S = 2.0           # window in which jaw / blink is expected
GAP_S = 1.5           # pause between trials
FIT_SLACK_S = 0.5     # extra time after the cue window used for fitting (late reactions)
JAW_RELAX_AT = 1.0    # "relax jaw" text after the cue (s)
LOOK_SILENCE_S = 1.0  # silence between the end of the go sound and the "eyes back" beep
LOOK_AFTER_S = 1.2    # time after the "eyes back" beep to finish the return movement
GO_SOUND = {"LOOK_LEFT": "go", "LOOK_RIGHT": "go"}

INSTRUCTIONS = {
    "REST": "Sit still, eyes on the screen, blink normally.",
    "JAW_CLENCH": "On '>>> NOW': clench your teeth.  On 'relax jaw': release.",
    "DOUBLE_BLINK": "On '>>> NOW': blink twice, quickly and firmly.",
    "LOOK_LEFT": "Keep your head still. Sound: eyes far LEFT (big movement) and hold. "
                 "Low beep: eyes back to the screen.",
    "LOOK_RIGHT": "Keep your head still. Sound: eyes far RIGHT (big movement) and hold. "
                  "Low beep: eyes back to the screen.",
}


@dataclass
class Marker:
    label: str
    onset: int      # sample
    dur: int        # samples


# --------------------------------------------------------------------------- protocol
def look_back_at(sound=None) -> float:
    """Seconds from the look go-cue to the 'eyes back' beep: go sound length + 1 s silence."""
    go = sound.duration("go") if sound is not None else 0.2
    return round(go + LOOK_SILENCE_S, 3)


def build_schedule(gestures=GESTURES, n_trials=8, rest_s=REST_S, back_at=1.2):
    """Returns (ui_messages [dict], markers, total_seconds). back_at: see look_back_at()."""
    def cue_len(g):
        return back_at + LOOK_AFTER_S if g.startswith("LOOK") else CUE_S

    msgs, markers = [], []
    blocks = (["REST"] if rest_s > 0 else []) + list(gestures)
    t = LEAD_S
    for b, g in enumerate(blocks, 1):
        if g == "REST":
            msgs.append(dict(t=0.0, kind="block", label=g, block=b, nblocks=len(blocks), secs=rest_s))
            markers.append(Marker("REST", int(t * FS), int(rest_s * FS)))
            t += rest_s
            continue
        msgs.append(dict(t=t, kind="block", label=g, block=b, nblocks=len(blocks),
                         secs=BLOCK_PAUSE_S + n_trials * (cue_len(g) + GAP_S)))
        t += BLOCK_PAUSE_S
        for i in range(1, n_trials + 1):
            msgs.append(dict(t=t, kind="cue_on", label=g, trial=i, n=n_trials, hold=back_at))
            if g.startswith("LOOK"):
                msgs.append(dict(t=t + back_at, kind="cue_back", label=g, trial=i, n=n_trials))
            elif g == "JAW_CLENCH":
                msgs.append(dict(t=t + JAW_RELAX_AT, kind="cue_back", label=g, trial=i, n=n_trials))
            markers.append(Marker(g, int(t * FS), int(cue_len(g) * FS)))
            t += cue_len(g)
            msgs.append(dict(t=t, kind="cue_off", label=g, trial=i, n=n_trials))
            t += GAP_S
    msgs.append(dict(t=t, kind="done", label=""))
    msgs.sort(key=lambda m: m["t"])
    return msgs, markers, t


class TerminalCueUI:
    """Cue display in the console. A GUI can implement the same show()/ask()."""

    def show(self, m: dict) -> None:
        k, g = m["kind"], m["label"].replace("_", " ")
        if k == "block":
            print(f"\n[Block {m['block']}/{m['nblocks']}] {g}  (~{m['secs']:.0f} s)")
            print(f"    {INSTRUCTIONS[m['label']]}")
        elif k == "cue_on":
            print(f"    {g:12s} {m['trial']}/{m['n']}   >>> NOW <<<", flush=True)
        elif k == "cue_back":
            msg = "<<< eyes back to screen" if m["label"].startswith("LOOK") else "relax jaw"
            print(f"                      {msg}", flush=True)
        elif k == "done":
            print("\nRecording finished.")

    def ask(self, question: str) -> bool:
        try:
            return input(question).strip().lower() in ("", "y", "yes")
        except EOFError:
            return False


class QuietUI:
    def __init__(self, redo: bool = False):
        self.redo = redo

    def show(self, m):
        pass

    def ask(self, question):
        r, self.redo = self.redo, False
        return r


def record_calibration(source, gestures=GESTURES, n_trials=8, rest_s=REST_S, ui=None, sound=None):
    ui = ui or TerminalCueUI()
    sound = sound or Sound(False)
    msgs, markers, total = build_schedule(gestures, n_trials, rest_s, look_back_at(sound))
    print(f"\nCalibration: {len(gestures)} gestures x {n_trials} trials"
          + (f" + {rest_s:.0f} s rest" if rest_s else "") + f"  -> about {total / 60:.1f} min")
    need = int(total * FS)
    source.flush()
    chunks, n, mi = [], 0, 0
    while n < need:
        while mi < len(msgs) and msgs[mi]["t"] * FS <= n:
            m = msgs[mi]
            ui.show(m)
            look = m["label"].startswith("LOOK")
            if m["kind"] == "cue_on":
                if look:                                   # sound only for look trials
                    sound.play(GO_SOUND[m["label"]])
                source.cue(m["label"], hold=m["hold"])
            elif m["kind"] == "cue_back" and look:
                sound.play("back")
            mi += 1
        x = source.read()
        if len(x):
            chunks.append(x)
            n += len(x)
    for m in msgs[mi:]:
        ui.show(m)
    sound.play("done")
    return np.concatenate(chunks)[:need], markers


def save_recording(path, raw, markers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, raw=raw, fs=FS, channels=np.array(CHANNELS + (("GYRO_SPEED",) if raw.shape[1] > 4 else ())),
        labels=np.array([m.label for m in markers]),
        onsets=np.array([m.onset for m in markers]),
        durs=np.array([m.dur for m in markers]),
    )


def load_recording(path):
    d = np.load(path, allow_pickle=False)
    markers = [Marker(str(lab), int(o), int(du)) for lab, o, du in zip(d["labels"], d["onsets"], d["durs"])]
    return d["raw"], markers


# --------------------------------------------------------------------------- fitting helpers
def _windows(markers):
    """(label, start, end): cue windows (+ slack) and 2.5 s windows cut from rest."""
    out = []
    slack = int(FIT_SLACK_S * FS)
    for m in markers:
        if m.label == "REST":
            w = int((CUE_S + FIT_SLACK_S) * FS)
            for s in range(m.onset, m.onset + m.dur - w + 1, w):
                out.append(("REST", s, s + w))
        else:
            out.append((m.label, m.onset, m.onset + m.dur + slack))
    return out


def fit_threshold(pos, neg, fp_weight=2.0):
    """Threshold that best separates positive from negative window statistics.

    Returns (threshold, separation) where separation = 10th pct of positives /
    95th pct of negatives (> 1 means the classes don't overlap much).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float(np.median(pos)) * 0.6 if pos.size else 1.0, float("nan")
    cands = np.unique(np.concatenate([pos, neg]))
    best, best_score = cands[0], -np.inf
    for c in cands:
        score = np.mean(pos >= c) - fp_weight * np.mean(neg >= c)
        if score > best_score:
            best, best_score = c, score
    below = neg[neg < best]
    lo = below.max() if below.size else best * 0.6
    thr = float(np.sqrt(max(lo, 1e-6) * best))        # geometric midpoint of the gap
    sep = float(np.percentile(pos, 10) / max(np.percentile(neg, 95), 1e-6))
    return thr, sep


def _pair_score(x, min_gap, max_gap, fs=FS):
    """Strongest blink PAIR in x: max over peak pairs min_gap..max_gap apart of the weaker peak."""
    pk, props = find_peaks(x, height=0, distance=int(0.12 * fs), width=(int(0.04 * fs), int(0.6 * fs)))
    t, h = pk / fs, props["peak_heights"]
    best = 0.0
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            if t[j] - t[i] > max_gap:
                break
            if t[j] - t[i] >= min_gap:
                best = max(best, float(min(h[i], h[j])))
    return best


def _second_peak(x, fs=FS):
    pk, props = find_peaks(x, height=0, distance=int(0.12 * fs))
    if len(pk) < 2:
        return 0.0, None
    order = np.argsort(props["peak_heights"])[::-1]
    a, b = sorted(pk[order[:2]])
    return float(props["peak_heights"][order[1]]), (b - a) / fs


def _longest_run(mask):
    best = run = 0
    for b in mask:
        run = run + 1 if b else 0
        best = max(best, run)
    return best


def _reject_trials(raw, f, wins, motion_thr, blink_w):
    """Indices of windows spoiled by head movement / electrode shifts / data gaps."""
    # slow artifact on the ear sensors, after removing the blink pattern (blinks are big there)
    ub = np.asarray(blink_w, float) / np.linalg.norm(blink_w)
    resid = f.eye - np.outer(f.eye @ ub, ub)
    low_tp = np.abs(resid[:, [CH["TP9"], CH["TP10"]]]).max(1)
    score = np.array([low_tp[s:e].max() for _, s, e in wins])
    has_motion = raw.shape[1] > 4
    rejected = {}
    for lab in set(w[0] for w in wins):
        idx = [i for i, w in enumerate(wins) if w[0] == lab]
        med = np.median(score[idx])
        for i in idx:
            _, s, e = wins[i]
            if score[i] > max(4 * med, 80.0):
                rejected[i] = "electrode shift"
            elif _longest_run(f.bad[s:e]) >= int(0.1 * FS):
                rejected[i] = "data gap"
            elif has_motion and np.nanmax(raw[s:e, 4]) > 2 * motion_thr:
                rejected[i] = "head movement"
    return rejected


# --------------------------------------------------------------------------- fitting
def fit_profile(raw, markers, base: Profile | None = None) -> Profile:
    p = Profile(**{**(base.__dict__ if base else {}), "report": {}})
    wins = _windows(markers)
    raw4 = raw[:, :4]
    f0 = FeatureExtractor(FS, p.line_freq, p.blink_w, p.gaze_w).process(raw4)

    # --- head-motion threshold (only with a gyroscope stream)
    if raw.shape[1] > 4:
        sp = np.nan_to_num(raw[:, 4])
        rest = np.concatenate([sp[s:e] for lab, s, e in wins if lab == "REST"] or [sp])
        looks = [sp[s:e].max() for lab, s, e in wins if lab.startswith("LOOK")]
        p.motion_thr = float(max(3 * np.percentile(rest, 99), 1.5 * np.median(looks) if looks else 0, 5.0))

    rejected = _reject_trials(raw, f0, wins, p.motion_thr, p.blink_w)
    keep = [w for i, w in enumerate(wins) if i not in rejected]
    by = lambda lab: [(s, e) for l, s, e in keep if l == lab]                         # noqa: E731
    others = lambda lab, excl=(): [(s, e) for l, s, e in keep if l != lab and l not in excl]  # noqa: E731
    report = {}

    # --- learn the 4-channel patterns of a blink and of a LEFT look (0.5-10 Hz, causal
    #     like live). Only accepted trials are used; natural blinks inside look trials are
    #     projected out before picking the look spike.
    eye = f0.eye
    vecs = []
    for s, e in by("DOUBLE_BLINK"):
        en = np.linalg.norm(eye[s:e], axis=1)
        pk, _ = find_peaks(en, distance=int(0.12 * FS))
        for q in pk[np.argsort(en[pk])[::-1][:2]]:
            vecs.append(eye[s + q])
    if len(vecs) >= 3:
        V = np.array(vecs)
        w = np.linalg.svd(V, full_matrices=False)[2][0]
        p.blink_w = (w if np.median(V @ w) >= 0 else -w).round(4).tolist()
    ub = np.asarray(p.blink_w) / np.linalg.norm(p.blink_w)
    outs = []
    for lab, sg in (("LOOK_LEFT", 1.0), ("LOOK_RIGHT", -1.0)):
        for s, e in by(lab):
            e_out = s + max(int(((e - s) / FS - LOOK_AFTER_S - FIT_SLACK_S) * FS), FS // 2)  # before back beep
            seg = eye[s:e_out]
            seg = seg - np.outer(seg @ ub, ub)                     # ignore blinks
            q = int(np.argmax(np.linalg.norm(seg, axis=1)))
            outs.append(sg * eye[s + q])
    if len(outs) >= 3:
        O = np.array(outs)
        w = np.linalg.svd(O, full_matrices=False)[2][0]
        p.gaze_w = (w if np.median(O @ w) >= 0 else -w).round(4).tolist()
    p.gaze_left_sign = 1

    f = FeatureExtractor(FS, p.line_freq, p.blink_w, p.gaze_w).process(raw4)

    # --- jaw: highest envelope level held for jaw_min_dur
    held = minimum_filter1d(f.emg, size=int(p.jaw_min_dur * FS))
    stat = lambda s, e: float(held[s:e].max())                                       # noqa: E731
    pos = [stat(s, e) for s, e in by("JAW_CLENCH")]
    neg = [stat(s, e) for s, e in others("JAW_CLENCH")]
    p.jaw_thr, sep = fit_threshold(pos, neg)
    report["JAW_CLENCH"] = {"thr": round(p.jaw_thr, 2), "separation": round(sep, 2)}

    # --- blink threshold: the 2nd-highest peak (= weaker blink of the pair).
    #     Rest/look windows contain natural blinks -> single blinks are negatives.
    #     First learn your double-blink timing from the two strongest peaks of each trial,
    #     then score every window by its strongest pair WITH that timing, so two natural
    #     blinks 1 s apart in a rest window don't count as a double blink.
    bh = f.v_up
    gaps = [g for g in (_second_peak(bh[s:e])[1] for s, e in by("DOUBLE_BLINK")) if g is not None]
    if gaps:
        p.blink_max_gap = float(np.clip(np.percentile(gaps, 90) * 1.4, 0.4, 0.8))
    pair = lambda s, e: _pair_score(bh[s:e], p.blink_min_gap, p.blink_max_gap)        # noqa: E731
    pos = [pair(s, e) for s, e in by("DOUBLE_BLINK")]
    neg = [pair(s, e) for s, e in others("DOUBLE_BLINK", excl=("JAW_CLENCH",))]
    p.blink_thr, sep = fit_threshold(pos, neg, fp_weight=1.0)
    # Natural quick blink pairs during rest can be as strong as deliberate ones; never let
    # them push the threshold above most of your real double blinks.
    if pos:
        p.blink_thr = float(min(p.blink_thr, 0.8 * np.percentile(pos, 20)))
    report["DOUBLE_BLINK"] = {"thr": round(p.blink_thr, 2), "separation": round(sep, 2),
                              "max_gap": round(p.blink_max_gap, 2),
                              "pattern": [round(v, 2) for v in p.blink_w]}

    # --- gaze: the smaller of the "out" and "back" spikes must exceed threshold
    stat = lambda s, e: float(min(f.hd[s:e].max(), -f.hd[s:e].min()))               # noqa: E731
    pos = [stat(s, e) for s, e in by("LOOK_LEFT") + by("LOOK_RIGHT")]
    neg = [stat(s, e) for s, e in others("LOOK_LEFT", excl=("LOOK_RIGHT", "JAW_CLENCH", "DOUBLE_BLINK"))]
    p.gaze_thr, sep = fit_threshold(pos, neg)
    if pos:                                   # keep it below most of your real look spikes
        p.gaze_thr = float(min(p.gaze_thr, 0.85 * np.percentile(pos, 20)))

    def first_sign(s, e):
        seg = f.hd[s:e]
        return np.sign(seg[np.argmax(np.abs(seg) > 0.5 * np.abs(seg).max())])

    gaps = []
    for s, e in by("LOOK_LEFT") + by("LOOK_RIGHT"):
        seg = f.hd[s:e]
        gaps.append(abs(int(np.argmax(seg)) - int(np.argmin(seg))) / FS)
    if gaps:
        p.gaze_max_gap = float(np.clip(np.percentile(gaps, 90) * 1.25, 1.6, 3.5))

    left = [first_sign(s, e) for s, e in by("LOOK_LEFT")]
    right = [first_sign(s, e) for s, e in by("LOOK_RIGHT")]
    cons = np.mean([v == 1 for v in left] + [v == -1 for v in right] or [0])
    report["LOOK_LEFT_RIGHT"] = {"thr": round(p.gaze_thr, 2), "separation": round(sep, 2),
                                 "max_gap": round(p.gaze_max_gap, 2),
                                 "direction_consistency": round(float(cons), 2),
                                 "pattern": [round(v, 2) for v in p.gaze_w]}

    counts = {}
    for i, (lab, _, _) in enumerate(wins):
        c = counts.setdefault(lab, {"used": 0, "rejected": 0, "reasons": {}})
        if i in rejected:
            c["rejected"] += 1
            c["reasons"][rejected[i]] = c["reasons"].get(rejected[i], 0) + 1
        else:
            c["used"] += 1
    p.calibrated = True
    p.report = {"fit": report, "trials": counts,
                "rejected_windows": [[wins[i][0], int(wins[i][1])] for i in sorted(rejected)],
                "motion_thr": round(p.motion_thr, 2) if raw.shape[1] > 4 else None}
    return p


# --------------------------------------------------------------------------- evaluation
def evaluate(raw, markers, profile: Profile) -> dict:
    """Replay the recording through the live engine and score the accepted trials."""
    events, _ = CommandEngine(profile).process(raw)
    rejected = {(lab, s) for lab, s in profile.report.get("rejected_windows", [])}
    labels = list(GESTURES)
    confusion = {g: {h: 0 for h in labels + ["none"]} for g in labels}
    slack = int(1.0 * FS)
    rest_s, rest_fp = 0.0, 0
    for lab, s, e in _windows(markers):
        if (lab, s) in rejected:
            continue
        if lab == "REST":
            rest_s += (e - s) / FS
            rest_fp += sum(1 for ev in events if s <= ev.t * FS < e)
            continue
        e = e + slack - int(FIT_SLACK_S * FS)
        hits = [ev for ev in events if s <= ev.t * FS <= e]
        confusion[lab][hits[0].gesture if hits else "none"] += 1
    n_trials = sum(sum(v.values()) for v in confusion.values())
    correct = sum(confusion[g][g] for g in labels)
    return {
        "accuracy": round(correct / max(n_trials, 1), 3),
        "per_gesture": {g: round(confusion[g][g] / max(sum(confusion[g].values()), 1), 3) for g in labels},
        "confusion": confusion,
        "false_positives_per_min_at_rest": round(rest_fp / max(rest_s, 1e-6) * 60, 2),
    }


def weak_gestures(p: Profile, ev: dict, n_trials: int) -> list[str]:
    weak = []
    trials = p.report["trials"]
    for g in GESTURES:
        used = trials.get(g, {}).get("used", 0)
        if ev["per_gesture"][g] < 0.75 or used < max(3, n_trials // 2):
            weak.append(g)
    return weak


def print_report(p: Profile, ev: dict) -> None:
    print("\n--- Calibration fit ---")
    for g, r in p.report["fit"].items():
        accs = [ev["per_gesture"][k] for k in GESTURES if k == g or (g == "LOOK_LEFT_RIGHT" and k.startswith("LOOK"))]
        flag = "" if min(accs) >= 0.75 else "   <-- weak: do this gesture bigger / more clearly"
        print(f"{g:16s} threshold={r['thr']:8.2f}  separation={r['separation']:5.2f}{flag}")
    lr = p.report["fit"]["LOOK_LEFT_RIGHT"]
    if lr["direction_consistency"] < 0.8:
        print("  warning: left/right direction was inconsistent; keep your head still and move eyes further")
    rej = {g: c for g, c in p.report["trials"].items() if c["rejected"]}
    if rej:
        print("rejected (movement / gaps): " + ", ".join(
            f"{g} {c['rejected']} ({', '.join(f'{k}' for k in c['reasons'])})" for g, c in rej.items()))
    if p.report.get("motion_thr") is not None:
        print(f"head-motion threshold: {p.report['motion_thr']} deg/s")
    print("\n--- Replay evaluation (accepted trials) ---")
    print(f"accuracy {ev['accuracy'] * 100:.1f}%   per gesture: " +
          ", ".join(f"{g}={v * 100:.0f}%" for g, v in ev["per_gesture"].items()))
    print(f"false positives at rest: {ev['false_positives_per_min_at_rest']}/min")
    if ev["false_positives_per_min_at_rest"] > 2:
        print("  note: during the rest period you blinked in quick pairs, which looks like a double blink.\n"
              "  Blink normally (relaxed) during rest; if this keeps happening, use a firmer double blink.")
    print("confusion (row = cued, col = detected first):")
    cols = list(GESTURES) + ["none"]
    print(" " * 14 + "".join(f"{c[:10]:>11s}" for c in cols))
    for g, row in ev["confusion"].items():
        print(f"{g:14s}" + "".join(f"{row[c]:>11d}" for c in cols))


def calibrate(source, profile_path, recording_dir="recordings", n_trials=8, rest_s=REST_S,
              ui=None, sound=None, max_redo=2) -> Profile:
    ui = ui or TerminalCueUI()
    sound = sound if sound is not None else Sound(True)
    raw, markers = record_calibration(source, GESTURES, n_trials, rest_s, ui, sound)
    for attempt in range(max_redo + 1):
        p = fit_profile(raw, markers)
        ev = evaluate(raw, markers, p)
        print_report(p, ev)
        weak = weak_gestures(p, ev, n_trials)
        if not weak or attempt == max_redo:
            break
        secs = build_schedule(weak, n_trials, 0, look_back_at(sound))[2]
        if not ui.ask(f"\nRedo only {', '.join(w.replace('_', ' ') for w in weak)} (~{secs:.0f} s)? [Y/n] "):
            break
        raw2, m2 = record_calibration(source, weak, n_trials, 0, ui, sound)
        if raw2.shape[1] != raw.shape[1]:
            print("stream changed (gyroscope on/off) - can't merge; keeping the first recording")
            break
        markers = [m for m in markers if m.label not in weak] + \
                  [Marker(m.label, m.onset + len(raw), m.dur) for m in m2]
        raw = np.concatenate([raw, raw2])

    rec_path = Path(recording_dir) / f"calib_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    save_recording(rec_path, raw, markers)
    p.report["evaluation"] = ev
    p.report["recording"] = str(rec_path)
    p.save(profile_path)
    print(f"\nprofile saved to {profile_path}\nrecording saved to {rec_path}")
    return p
