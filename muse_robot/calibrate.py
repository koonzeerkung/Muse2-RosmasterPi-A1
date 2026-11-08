"""Calibration: cue the user on screen, record, fit per-user thresholds, evaluate.

Protocol (default 8 trials, about 1.9 min):
  15 s rest (eyes open, blink normally)
  3 blocks (jaw clench, double blink, eyebrow raise) of N trials,
  each trial = 2.0 s cue window + 1.5 s pause; cues are on screen only

Robustness
  * Natural blinks are expected everywhere: rest data with natural blinks teaches
    the double-blink threshold to ignore single blinks and blinks far apart.
  * Trials spoiled by head movement / big electrode shifts / Bluetooth gaps are
    detected and left out of the threshold fitting (reported as "rejected").
  * Gestures that still come out weak can be re-recorded on their own (redo).

The raw recording + cue markers are saved as .npz so you can re-fit later or use
them to train an ML model. Older recordings that also contain LOOK_LEFT/LOOK_RIGHT
trials still work: those trials are used as "not this gesture" examples.
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
CUE_S = 2.0           # window in which the gesture is expected
GAP_S = 1.5           # pause between trials
FIT_SLACK_S = 0.5     # extra time after the cue window used for fitting (late reactions)
RELAX_AT = 1.0        # "relax" text after the cue for jaw / eyebrow (s)

INSTRUCTIONS = {
    "REST": "Sit still, eyes on the screen, blink normally.",
    "JAW_CLENCH": "On '>>> NOW': clench your teeth.  On 'relax jaw': release.",
    "DOUBLE_BLINK": "On '>>> NOW': blink twice, quickly and firmly.",
    "EYEBROW_RAISE": "On '>>> NOW': raise your eyebrows high (teeth apart).  On 'relax': release.",
}


@dataclass
class Marker:
    label: str
    onset: int      # sample
    dur: int        # samples


# --------------------------------------------------------------------------- protocol
def build_schedule(gestures=GESTURES, n_trials=8, rest_s=REST_S):
    """Returns (ui_messages [dict], markers, total_seconds)."""
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
                         secs=BLOCK_PAUSE_S + n_trials * (CUE_S + GAP_S)))
        t += BLOCK_PAUSE_S
        for i in range(1, n_trials + 1):
            msgs.append(dict(t=t, kind="cue_on", label=g, trial=i, n=n_trials))
            if g in ("JAW_CLENCH", "EYEBROW_RAISE"):
                msgs.append(dict(t=t + RELAX_AT, kind="cue_back", label=g, trial=i, n=n_trials))
            markers.append(Marker(g, int(t * FS), int(CUE_S * FS)))
            t += CUE_S
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
            print(f"                      relax {'jaw' if m['label'] == 'JAW_CLENCH' else 'eyebrows'}",
                  flush=True)
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
    msgs, markers, total = build_schedule(gestures, n_trials, rest_s)
    print(f"\nCalibration: {len(gestures)} gestures x {n_trials} trials"
          + (f" + {rest_s:.0f} s rest" if rest_s else "") + f"  -> about {total / 60:.1f} min")
    need = int(total * FS)
    source.flush()
    chunks, n, mi = [], 0, 0
    while n < need:
        while mi < len(msgs) and msgs[mi]["t"] * FS <= n:
            m = msgs[mi]
            ui.show(m)
            if m["kind"] == "cue_on":
                source.cue(m["label"])
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
    f0 = FeatureExtractor(FS, p.line_freq, p.blink_w).process(raw4)

    # --- head-motion threshold (only with a gyroscope stream): above the stillness at rest
    #     and above what the user does during the gesture trials
    if raw.shape[1] > 4:
        sp = np.nan_to_num(raw[:, 4])
        rest = np.concatenate([sp[s:e] for lab, s, e in wins if lab == "REST"] or [sp])
        trials = [sp[s:e].max() for lab, s, e in wins if lab in GESTURES]
        p.motion_thr = float(max(3 * np.percentile(rest, 99), 1.5 * np.median(trials) if trials else 0, 5.0))

    rejected = _reject_trials(raw, f0, wins, p.motion_thr, p.blink_w)
    keep = [w for i, w in enumerate(wins) if i not in rejected]
    by = lambda lab: [(s, e) for l, s, e in keep if l == lab]                         # noqa: E731
    others = lambda lab, excl=(): [(s, e) for l, s, e in keep if l != lab and l not in excl]  # noqa: E731
    report = {}

    # --- learn the 4-channel pattern of a blink (0.5-10 Hz, causal like live) from the
    #     two strongest deflections of each accepted double-blink trial
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

    f = FeatureExtractor(FS, p.line_freq, p.blink_w).process(raw4)

    # --- muscle burst (jaw or eyebrow): highest envelope level held for jaw_min_dur
    n_min = int(p.jaw_min_dur * FS)
    held = minimum_filter1d(f.emg, size=n_min)
    stat = lambda s, e: float(held[s:e].max())                                       # noqa: E731
    muscle = ("JAW_CLENCH", "EYEBROW_RAISE")
    pos = [stat(s, e) for s, e in by("JAW_CLENCH") + by("EYEBROW_RAISE")]
    # negatives: rest and blink trials only (eyebrow trials are muscle bursts too; trials of
    # gestures this version doesn't use, e.g. look in old recordings, must not raise it)
    neg = [stat(s, e) for l, s, e in keep if l in ("REST", "DOUBLE_BLINK")]
    p.jaw_thr, sep = fit_threshold(pos, neg)
    report["JAW_CLENCH"] = {"thr": round(p.jaw_thr, 2), "separation": round(sep, 2)}

    # --- jaw vs eyebrow: how far the forehead signal (AF7/AF8, 0.5-10 Hz) moves between the
    #     start of the burst and brow_decide_s later - exactly what the live detector measures.
    #     Raising the eyebrows moves the forehead skin a lot; clenching the jaw barely does.
    fore_slow = (f.eye[:, CH["AF7"]] + f.eye[:, CH["AF8"]]) / 2
    dec_n = int(p.brow_decide_s * FS)
    above = (f.emg > p.jaw_thr).astype(int)
    run_ok = np.convolve(above, np.ones(n_min, int), "full")[n_min - 1:] == n_min   # n_min above from i

    def shifts(s, e):
        idx = np.where(run_ok[s:e])[0]
        if len(idx) == 0:
            return None
        a = s + idx[0]
        base = fore_slow[max(a - 13, 0):a].mean() if a > 0 else fore_slow[a]
        seg = fore_slow[a:a + dec_n + 1] - base
        return float(seg.max()), float(-seg.min())                     # (up, down)

    jaw_s = [x for x in (shifts(s, e) for s, e in by("JAW_CLENCH")) if x is not None]
    brow_s = [x for x in (shifts(s, e) for s, e in by("EYEBROW_RAISE")) if x is not None]
    p.brow_shift_thr, brow_sep, jaw_x, brow_x = 1e9, float("nan"), [], []
    if len(jaw_s) >= 3 and len(brow_s) >= 3:
        p.brow_dir = 1 if np.median([u for u, _ in brow_s]) > np.median([d for _, d in brow_s]) else -1
        k = 0 if p.brow_dir > 0 else 1
        jaw_x = [v[k] for v in jaw_s]
        brow_x = [v[k] for v in brow_s]
        if np.median(brow_x) > np.median(jaw_x):
            cands = np.unique(np.concatenate([jaw_x, brow_x]))
            # most trials right; ties -> the higher boundary
            score = [np.mean(np.array(brow_x) > c) + np.mean(np.array(jaw_x) <= c) for c in cands]
            c = cands[max(range(len(cands)), key=lambda i: (round(score[i], 6), cands[i]))]
            higher = [v for v in brow_x if v > c]
            # middle of the gap: eyebrow shifts are much larger than jaw shifts, and a jaw clench
            # can move the forehead more on another day, so keep the line well above the jaw side
            p.brow_shift_thr = float((c + min(higher)) / 2) if higher else float(c)
            brow_sep = float(np.percentile(brow_x, 10) / max(np.percentile(jaw_x, 90), 1e-6))
    report["EYEBROW_RAISE"] = {"thr": round(p.brow_shift_thr, 1), "separation": round(brow_sep, 2),
                               "direction": p.brow_dir, "decide_s": p.brow_decide_s,
                               "jaw_shift": [round(v, 1) for v in jaw_x],
                               "brow_shift": [round(v, 1) for v in brow_x]}

    # --- double blink: first learn the user's double-blink timing from the two strongest
    #     peaks of each trial, then score every window by its strongest blink PAIR with that
    #     timing, so two natural blinks 1 s apart in a rest window don't count.
    bh = f.v_up
    gaps = [g for g in (_second_peak(bh[s:e])[1] for s, e in by("DOUBLE_BLINK")) if g is not None]
    if gaps:
        p.blink_max_gap = float(np.clip(np.percentile(gaps, 90) * 1.4, 0.4, 0.8))
    pair = lambda s, e: _pair_score(bh[s:e], p.blink_min_gap, p.blink_max_gap)        # noqa: E731
    pos = [pair(s, e) for s, e in by("DOUBLE_BLINK")]
    neg = [pair(s, e) for s, e in others("DOUBLE_BLINK", excl=muscle)]
    p.blink_thr, sep = fit_threshold(pos, neg, fp_weight=1.0)
    # Natural quick blink pairs during rest can be as strong as deliberate ones; never let
    # them push the threshold above most of the user's real double blinks.
    if pos:
        p.blink_thr = float(min(p.blink_thr, 0.8 * np.percentile(pos, 20)))
    report["DOUBLE_BLINK"] = {"thr": round(p.blink_thr, 2), "separation": round(sep, 2),
                              "max_gap": round(p.blink_max_gap, 2),
                              "pattern": [round(v, 2) for v in p.blink_w]}

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
    """Replay the recording through the live engine and score the accepted trials.

    Windows of other labels (e.g. LOOK trials in old recordings) count as
    "nothing should fire here" together with the rest windows.
    """
    events, _ = CommandEngine(profile).process(raw)
    rejected = {(lab, s) for lab, s in profile.report.get("rejected_windows", [])}
    labels = list(GESTURES)
    confusion = {g: {h: 0 for h in labels + ["unclear", "none"]} for g in labels}
    slack = int(1.0 * FS)
    rest_s, rest_fp, other_fp = 0.0, 0, 0
    for lab, s, e in _windows(markers):
        if (lab, s) in rejected:
            continue
        if lab == "REST":
            rest_s += (e - s) / FS
            rest_fp += sum(1 for ev in events if s <= ev.t * FS < e)
            continue
        e = e + slack - int(FIT_SLACK_S * FS)
        early = s - int(0.3 * FS)             # people sometimes start just before the cue
        hits = [ev for ev in events if early <= ev.t * FS <= e]
        if lab in confusion:
            first = hits[0].gesture if hits else "none"
            confusion[lab][first if first in confusion[lab] else "unclear"] += 1
        else:
            other_fp += len(hits)
    n_trials = sum(sum(v.values()) for v in confusion.values())
    correct = sum(confusion[g][g] for g in labels)
    return {
        "accuracy": round(correct / max(n_trials, 1), 3),
        "per_gesture": {g: round(confusion[g][g] / max(sum(confusion[g].values()), 1), 3) for g in labels},
        "confusion": confusion,
        "false_positives_per_min_at_rest": round(rest_fp / max(rest_s, 1e-6) * 60, 2),
        "false_events_in_other_trials": other_fp,
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
    names = {"JAW_CLENCH": "muscle burst", "DOUBLE_BLINK": "double blink", "EYEBROW_RAISE": "brow shift (uV)"}
    for g, r in p.report["fit"].items():
        flag = "" if ev["per_gesture"].get(g, 1) >= 0.75 else "   <-- weak: do this gesture bigger / more clearly"
        print(f"{names.get(g, g):16s} threshold={r['thr']:8.2f}  separation={r['separation']:5.2f}{flag}")
    rej = {g: c for g, c in p.report["trials"].items() if c["rejected"]}
    if rej:
        print("rejected (movement / gaps): " + ", ".join(
            f"{g} {c['rejected']} ({', '.join(f'{k}' for k in c['reasons'])})" for g, c in rej.items()))
    if p.report.get("motion_thr") is not None:
        print(f"head-motion threshold: {p.report['motion_thr']} deg/s")
    br = p.report["fit"].get("EYEBROW_RAISE")
    if br and br["thr"] < 1e8:
        print(f"  eyebrow raise ON: each muscle burst is decided {br['decide_s']} s after it starts; "
              f"bursts within +-{int(p.brow_margin * 100)}% of the threshold are ignored as unclear")
        close = [v for v in br["jaw_shift"] + br["brow_shift"]
                 if p.brow_shift_thr * (1 - p.brow_margin) < v < p.brow_shift_thr * (1 + p.brow_margin)]
        if close:
            print(f"  note: {len(close)} calibration trial(s) fell in the unclear zone - clench without moving\n"
                  "  the eyebrows, and raise the eyebrows higher with the teeth apart")
    if br and br["thr"] >= 1e8:
        print("  warning: eyebrow raise and jaw clench looked the same on your headset, so eyebrow raise\n"
              "  is OFF (every muscle burst = jaw clench). Raise the eyebrows higher with the teeth apart,\n"
              "  check the forehead pads sit flat, and redo the eyebrow block.")
    print("\n--- Replay evaluation (accepted trials) ---")
    print(f"accuracy {ev['accuracy'] * 100:.1f}%   per gesture: " +
          ", ".join(f"{g}={v * 100:.0f}%" for g, v in ev["per_gesture"].items()))
    print(f"false positives at rest: {ev['false_positives_per_min_at_rest']}/min")
    if ev["false_positives_per_min_at_rest"] > 2:
        print("  note: during the rest period you blinked in quick pairs, which looks like a double blink.\n"
              "  Blink normally (relaxed) during rest; if this keeps happening, use a firmer double blink.")
    print("confusion (row = cued, col = detected first):")
    cols = list(GESTURES) + ["unclear", "none"]
    print(" " * 14 + "".join(f"{c[:12]:>13s}" for c in cols))
    for g, row in ev["confusion"].items():
        print(f"{g:14s}" + "".join(f"{row[c]:>13d}" for c in cols))


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
        secs = build_schedule(weak, n_trials, 0)[2]
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
