# Muse 2 EEG → robot command packets

The EEG part of the Yahboom RosmasterPi A1 project. Three deliberate face gestures are
detected in real time from a **Muse 2** headband and sent as small JSON packets over Wi-Fi
(UDP) to the robot side, which runs the robot, the camera and the music.

| Gesture | Command | Toggles | State field |
|---|---|---|---|
| **Jaw clench** | `TOGGLE_RUN` | robot run ↔ stop | `run` |
| **Double blink** | `TOGGLE_MODE` | gaze controls the robot movement ↔ the camera | `mode` |
| **Eyebrow raise** | `TOGGLE_MUSIC` | music on ↔ off | `music` |

```
Muse 2 ──BLE──> BlueMuse ──LSL──> muse_robot (Windows laptop) ──UDP:9000──> robot side
                                  clean · detect · calibrate     JSON packets   (receives, drives
                                                                 + heartbeat     robot/camera/music)
```

**Robot-side developers: read [receiver/PROTOCOL.md](receiver/PROTOCOL.md)** - the packet format,
and [receiver/example_receiver.py](receiver/example_receiver.py), a standard-library receiver
with `on_run` / `on_mode` / `on_music` / `on_link_lost` functions to fill in.

---

## Contents

- [Requirements](#requirements)
- [Quick start (Windows)](#quick-start-windows)
- [The packets](#the-packets)
- [How it works](#how-it-works)
- [Calibration](#calibration)
- [Results](#results)
- [Command line](#command-line)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Requirements

| | |
|---|---|
| Headset | Interaxon **Muse 2** (TP9, AF7, AF8, TP10 at 256 Hz; gyroscope optional) |
| Laptop | Windows 10/11, Bluetooth, **Python 3.9+**, [BlueMuse](https://github.com/kowalej/BlueMuse) to stream the Muse over LSL |
| Network | the EEG laptop and the robot-side computer on the same network (UDP port 9000 by default) |
| Python packages | `numpy`, `scipy`, `pylsl`, `pyserial` - installed automatically by `MuseRobot.bat` (`brainflow` only for `--source brainflow`, install it yourself) |
| Robot side | anything that can receive UDP; `receiver/example_receiver.py` needs only Python 3.8+ |

## Quick start (Windows)

1. Pair the Muse 2, open **BlueMuse**, press **Start Streaming** (enable the Gyroscope stream
   too if your BlueMuse version offers it - head movements will then be ignored).
2. Double-click **`MuseRobot.bat`**. The first run creates a private `.venv` and installs the
   packages; after that it opens this menu:

   ```
   1. Check signal   (NaN, contact quality, 10 s)
   2. Calibrate      (about 2 minutes)
   3. Run            - show commands on screen only
   4. Run            - send packets to the robot side (UDP)
   5. Change profile name
   6. Test without headset (simulator)
   7. Send test packets (keyboard, no headset)
   8. Listen for packets (test receiver)
   ```

3. **1 - Check signal**: every channel should say `OK`. If not, push the headband down so the
   forehead pads sit flat, move hair away from the ear sensors and check again. (The ear
   sensors matter most for the jaw clench.)
4. **2 - Calibrate** (~2 min, prompts on screen). **Every person needs their own profile**, and
   recalibrating each time the headset is put on gives the best results.
5. **3 - Run** to see the commands on screen, then **4 - Run** with the robot-side computer's IP
   address. While running, `Ctrl+C` then `N` returns to the menu.

### Doing the gestures

- **Jaw clench** - bite down firmly for about a second, **eyebrows still**.
- **Double blink** - two quick, *firm* blinks (~0.3 s apart). Natural blinks that happen to
  come in a quick pair look the same, so this is the gesture that can trigger by accident.
- **Eyebrow raise** - raise both eyebrows high for about a second, **teeth apart**.

The screen shows every command with the state after it, e.g.
`JAW_CLENCH -> TOGGLE_RUN  run=ON  mode=robot  music=off  (#5)`.

## The packets

Full specification: **[receiver/PROTOCOL.md](receiver/PROTOCOL.md)**. In short:

- one JSON object per UDP datagram, port 9000 by default;
- an **event** message for every gesture, and a **heartbeat** every 0.5 s;
- every message carries the full **state** after it (`run`, `mode`, `music`), so the robot side
  can simply follow `state`: a lost packet is corrected by the next heartbeat;
- `seq` counts messages, so lost or late packets can be spotted;
- the robot side should **stop the robot if nothing arrives for 1.5 s**.

```json
{"v": 1, "type": "event", "seq": 42, "time": 1762610000.123, "source": "muse2",
 "event": {"gesture": "JAW_CLENCH", "command": "TOGGLE_RUN", "t": 123.45, "info": {}},
 "state": {"run": true, "mode": "robot", "music": false}}
```

Start state: `run=false`, `mode="robot"` (`--start-mode camera` to change), `music=false`.
The gesture → command mapping is `DEFAULT_COMMAND_MAP` in `muse_robot/config.py`.

**Testing without the robot:** menu **7** sends the three toggles from the keyboard (`j`, `b`,
`e`); menu **8** (or `receiver/example_receiver.py` on the robot-side computer) shows what arrives.

## How it works

### What the Muse 2 actually records

Measured on real recordings with this project:

- Muse 2's reference electrode (Fpz) sits on the forehead **right above the eyes**. A blink is
  therefore seen mostly as a large *negative* dip (≈ −250 µV) on the **ear** sensors TP9/TP10,
  and only as a small wiggle on the forehead sensors AF7/AF8.
- Clenching the jaw produces broadband muscle activity (EMG) far above the EEG band, strongest
  at the ear sensors - the most robust signal the headband gives.
- Raising the eyebrows also produces a muscle burst on all four sensors, and its EMG pattern
  overlaps with the jaw clench. But it **moves the forehead skin under AF7/AF8**: within 0.3 s
  the forehead signal shifts by 630–1580 µV, versus 3–143 µV for a jaw clench.

### Pipeline (`muse_robot/`)

1. **Clean** - NaN samples from Bluetooth dropouts are filled with the last good value (IIR
   filters would otherwise turn one NaN into NaN forever); 50/100 Hz mains notch.
2. **Features** - all filters are causal and keep state between chunks, so offline replay of a
   recording gives exactly the same events as live:
   - **EMG envelope**: 30–110 Hz band, RMS over 100 ms, averaged over the 4 channels.
   - **Forehead slow signal**: AF7/AF8 at 0.5–10 Hz, for jaw vs eyebrow.
   - **Blink signal B**: the 0.5–10 Hz signal projected on the 4-channel blink pattern learned
     in calibration, with the sideways-eye direction (AF7 − AF8) removed so glancing left or
     right does not look like a blink.
3. **Detectors** (`detectors.py`):
   - **Muscle burst**: envelope above threshold for ≥ 120 ms. 0.3 s after the burst started,
     the forehead shift decides: clearly above the learned threshold → `EYEBROW_RAISE`, clearly
     below → `JAW_CLENCH`, within ±25 % of the threshold → **unclear, ignored** (shown on screen,
     never sent - every command is a toggle, so a wrong guess would flip the wrong thing).
   - **Double blink**: two B peaks (40–600 ms wide) between 0.12 s and the learned max gap
     apart; a single blink just expires.
4. **Safety rules** - blink candidates overlapping a muscle burst or a head movement
   (gyroscope) are discarded; after a data gap ≥ 100 ms, blink events are muted for 0.6 s.
   One event per gesture: a held clench or raise counts once (0.8 s lockout between muscle
   events, 0.5 s between double blinks).

Latency from the start of the gesture: jaw clench / eyebrow raise ≈ 0.3 s, double blink ≈ 0.1 s
after the second blink.

Why not a neural network: calibration gives ~24 labelled trials per person, the signals are
tens to hundreds of µV (far larger than EEG), and a rule-based detector can explain every
decision. Every calibration is saved under `recordings/`, so the same data can train and
benchmark an ML model later.

## Calibration

```
15 s rest -> JAW CLENCH x8 -> DOUBLE BLINK x8 -> EYEBROW RAISE x8          (~1.9 min)
screen: "[Block 2/4] JAW CLENCH"  "JAW CLENCH 3/8 >>> NOW <<<"  ...  "relax jaw"
```

What calibration learns per person: the blink pattern, the muscle-burst and blink thresholds,
the forehead-shift threshold and direction for jaw vs eyebrow, and the double-blink timing.
Natural blinks and small movements are expected: rest data with natural blinks teaches the
threshold to ignore single blinks; trials spoiled by electrode shifts, Bluetooth gaps or head
movement are rejected and reported; a gesture that comes out weak (< 75 % replay accuracy) can
be re-recorded on its own. The report shows per-gesture accuracy, a confusion matrix (with an
`unclear` column) and false triggers per minute at rest. If eyebrow raises and jaw clenches
look the same on someone's headset, eyebrow raise is switched off and the report says so.

## Results

Real recordings with eyebrow trials: person z1 (two sessions, two days) and person z2 (one
session), 8 trials per gesture. Each session's profile was tested on its own recording and on
the others.

| | Jaw clench | Double blink | Eyebrow raise |
|---|---|---|---|
| Same session, z1 (2 sessions) | **100 %** | **100 %** | **100 %** |
| Same session, z2 | 86 % | 88 % | 75 % |
| Same person, other session (z1 ↔ z1) | 88–100 % | 88–100 % | 75–100 % |
| Jaw clench taken as eyebrow raise | **never** (0 in all 9 combinations) | | |

- False triggers at rest come from natural blinks in quick pairs (0–32 per minute depending on
  the person and session) - double blink is the gesture to watch.
- **Profiles don't transfer between people**: z1's profile on z2's recording took 6 of 8
  eyebrow raises as jaw clenches. Everyone calibrates their own profile.
- Earlier sessions without eyebrow trials (4 sessions): jaw clench 100 % in every same-session
  and cross-session combination.

**Simulator** (`python -m muse_robot sim-test`, signal model based on the real recordings, with
natural blinks, NaN dropouts and frequent head movements): 95–100 % of 80 gestures over three
seeds, every jaw clench and eyebrow raise correct, 0–2 false events in 6.8 min. The simulator
is a regression test, not a substitute for real data.

## Command line

```bash
python -m muse_robot check                                   # stream quality, NaN, contact (10 s)
python -m muse_robot calibrate --profile profiles/me.json    # --trials 5 for a quicker run
python -m muse_robot run --profile profiles/me.json --output udp:192.168.1.20:9000
python -m muse_robot run --profile profiles/me.json --output serial:COM5:115200
python -m muse_robot send-test --output udp:192.168.1.20:9000   # keyboard: j / b / e
python -m muse_robot listen --port 9000                      # show incoming packets
python -m muse_robot fit recordings/calib_XXXX.npz --profile profiles/me.json   # re-fit
python -m muse_robot sim-test                                # no hardware needed
python -m pytest tests                                       # tests
```

`--output` can be given several times; the screen output is always on. `--start-mode camera`
starts in camera mode, `--sound` beeps on every detected command. `--source brainflow`
connects to the Muse directly over Bluetooth instead of BlueMuse (`pip install brainflow`).

## Project structure

```
MuseRobot.bat              Windows menu (sets up Python on first run)
muse_robot/
  config.py                constants, gesture -> command map, per-person Profile (JSON)
  protocol.py              packet format, toggle state, sequence numbers
  outputs.py               screen, UDP, serial
  sources.py               BlueMuse/muselsl (LSL) + gyroscope, BrainFlow, simulator
  dsp.py                   NaN handling, filters, EMG envelopes, blink signal
  detectors.py             muscle-burst (jaw / eyebrow / unclear) and double-blink detectors
  calibrate.py             cue protocol, trial rejection, fitting, replay evaluation
  sounds.py                optional beep on each command (run --sound), end-of-calibration chime
  cli.py                   command line
receiver/
  PROTOCOL.md              packet specification for the robot side
  example_receiver.py      standard-library receiver to start the robot-side code from
tests/test_pipeline.py     pytest suite (filters, NaN, motion gating, packets, receiver, simulator)
requirements.txt           Python packages
lsl_api.cfg                keeps the LSL library quiet
```

`profiles/` (calibration results) and `recordings/` (raw EEG of calibration sessions) are
created locally and are **not** committed: they are personal biosignal data.

Branches: **`main`** is the final team design described here (three gestures → toggle
packets). `Jaw/DoubleBlink/Eyebrow` and `Jaw_DoubleBlink` keep earlier development stages
(the older GO/STOP commands and ROS 2 bridge).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No EEG stream found` | BlueMuse must be streaming (Start Streaming) |
| A channel is not `OK` in Check signal | reseat the headband; forehead pads flat, hair away from ear sensors |
| Jaw clench never detected | ear sensors (TP9/TP10) not touching: check signal, fix the fit, recalibrate |
| Eyebrow raise taken as jaw clench, or "ignored" | raise higher with the teeth apart; forehead pads flat; recalibrate |
| Jaw clench "ignored" | keep the eyebrows still while clenching |
| Random `TOGGLE_MODE` | natural quick double blinks; blink relaxed, use a firmer deliberate double blink |
| Robot side receives nothing | IP and port, same network, allow Python in Windows Firewall (private networks); test with menu 7 → 8 |

## References

- BlueMuse - Muse streaming over LSL on Windows: <https://github.com/kowalej/BlueMuse>
- Lab Streaming Layer / pylsl: <https://github.com/sccn/labstreaminglayer>
- muse-lsl: <https://github.com/alexandrebarachant/muse-lsl>
- Krigolson et al. (2017), *Choosing MUSE: Validation of a Low-Cost, Portable EEG System for ERP
  Research*, Frontiers in Neuroscience: <https://www.frontiersin.org/articles/10.3389/fnins.2017.00109>
