# Muse 2 → Yahboom RosmasterPi A1

Drive a **Yahboom RosmasterPi A1** robot car with a **Muse 2** EEG headband. Three deliberate
face gestures are detected in real time from the headband's 4 channels and sent over Wi-Fi
to the robot, which drives through ROS 2. (Steering by gaze is a separate, camera-based part
of the project.)

| Gesture | Detected from | Command | Robot |
|---|---|---|---|
| **Jaw clench** | muscle burst (EMG, 30–110 Hz), forehead sensors stay still | `STOP` | stops (always wins) |
| **Double blink** | two blink peaks ~0.2–0.5 s apart | `GO` | drives forward |
| **Eyebrow raise** | muscle burst + large slow shift of the forehead sensors | `BROW` | not used by the bridge yet - map it in `config.py` |

```
Muse 2 ──BLE──> BlueMuse ──LSL──> muse_robot (Windows laptop) ──Wi-Fi/UDP──> eeg_cmd_bridge.py ──/cmd_vel──> RosmasterPi A1
                                  clean · detect · calibrate        port 9000     (on the robot, ROS 2 Humble)
```

No ROS is needed on the laptop, and no GPU or deep-learning framework: the classifier is a
calibrated signal-processing pipeline that runs comfortably in real time on any laptop.

---

## Contents

- [Requirements](#requirements)
- [Quick start (Windows)](#quick-start-windows)
- [Robot setup](#robot-setup)
- [How it works](#how-it-works)
- [Calibration](#calibration)
- [Results](#results)
- [Command line](#command-line)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)
- [References](#references)

## Requirements

| | |
|---|---|
| Headset | Interaxon **Muse 2** (TP9, AF7, AF8, TP10 at 256 Hz; gyroscope optional) |
| Laptop | Windows 10/11, Bluetooth, **Python 3.9+**, [BlueMuse](https://github.com/kowalej/BlueMuse) to stream the Muse over LSL |
| Robot | Yahboom **RosmasterPi A1** (Raspberry Pi 5, ROS 2 Humble), on the same Wi-Fi as the laptop |
| Python packages | `numpy`, `scipy`, `pylsl` (+ `pyserial` for serial output, `brainflow` optional) - installed automatically |

## Quick start (Windows)

1. Pair the Muse 2, open **BlueMuse**, press **Start Streaming** (enable the Gyroscope stream
   too if your BlueMuse version offers it - head movements will then be ignored).
2. Double-click **`MuseRobot.bat`**. The first run creates a private `.venv` and installs the
   packages; after that it opens this menu:

   ```
   1. Check signal   (NaN, contact quality, 10 s)
   2. Calibrate      (about 2 minutes)
   3. Run            - show commands on screen only
   4. Run            - send to a SERIAL device (COM port)
   5. Run            - drive the YAHBOOM ROBOT over Wi-Fi
   6. Change profile name
   7. Test without headset (simulator)
   8. Test robot connection (keyboard, no headset)
   ```

3. **1 - Check signal**: every channel should say `OK`. If not, push the headband down so the
   forehead pads sit flat, move hair away from the ear sensors and check again.
4. **2 - Calibrate** (~2 min, prompts on screen). Recalibrate every time you put the
   headset on - the blink signal depends on how it sits.
5. **3 - Run** to see the commands on screen, then **5 - Run** to drive the robot
   (after the [robot setup](#robot-setup)). While running, `Ctrl+C` then `N` returns to the menu.

### Doing the gestures

- **Jaw clench** - bite down firmly for about a second. Very reliable, so it is `STOP`.
- **Double blink** - two quick, *firm* blinks (~0.3 s apart). Natural blinks that happen to
  come in a quick pair look the same, so this is the gesture that can trigger by accident.
- **Eyebrow raise** - raise both eyebrows high for about a second, **teeth apart**. An unclear
  burst always counts as `STOP`.

## Robot setup

Full guide: **[robot/README_robot.md](robot/README_robot.md)**. In short:

1. Robot and laptop on the same Wi-Fi; note the robot's IP address.
2. Copy the `robot/` folder to the robot, e.g.
   `scp -r robot pi@ROBOT_IP:~/eeg_bridge`.
3. On the robot: start the chassis driver as in Yahboom's A1 tutorial (`yahboomcar_bringup`,
   the same one used for keyboard/joystick control), check `ros2 topic info /cmd_vel`, then
   `bash ~/eeg_bridge/start_bridge.sh`.
4. On the laptop: **8 - Test robot connection** (`g` = GO, `s` = STOP, `a`/`d` = steer;
   wheels off the ground the first time), then **5 - Run**.

**Safety:** the laptop sends a heartbeat twice a second; if the robot hears nothing for 1.5 s
(program closed, laptop asleep, Wi-Fi lost) it stops by itself and waits for a new `GO`.
Closing the laptop program sends `STOP`. `STOP` is never blocked by any filter.
Speed (default 0.15 m/s) and timeouts are options of `start_bridge.sh`
(`--max-drive 20` stops automatically after 20 s without a new command).

## How it works

### What the Muse 2 actually records

Measured on real recordings with this project:

- Muse 2's reference electrode (Fpz) sits on the forehead **right above the eyes**. A blink is
  therefore seen mostly as a large *negative* dip (≈ −250 µV) on the **ear** sensors TP9/TP10,
  and only as a small wiggle on the forehead sensors AF7/AF8.
- Clenching the jaw produces broadband muscle activity (EMG) far above the EEG band - the
  most robust signal the headband gives. It reaches all four sensors, strongest at the ear
  sensors TP9/TP10.
- Raising the eyebrows also produces a muscle burst on all four sensors - on the real
  recording its forehead/ear EMG ratio overlapped with the jaw clench, so EMG alone can't tell
  them apart. But it **moves the forehead skin under AF7/AF8**: within 0.3 s the forehead
  signal shifts by 630–955 µV, versus 10–40 µV for a jaw clench (8 trials each, same session).

### Pipeline (`muse_robot/`)

1. **Clean** - NaN samples from Bluetooth dropouts are filled with the last good value (IIR
   filters would otherwise turn one NaN into NaN forever); 50/100 Hz mains notch.
2. **Features** - all filters are causal and keep state between chunks, so offline replay of a
   recording gives exactly the same events as live:
   - **EMG envelope**: 30–110 Hz band, RMS over 100 ms, averaged over the 4 channels (the
     ear and forehead pairs are also logged separately).
   - **Forehead slow signal**: AF7/AF8 at 0.5–10 Hz, for the eyebrow decision.
   - **Blink signal B**: the 0.5–10 Hz signal projected on the 4-channel blink pattern learned
     in calibration, with the sideways-eye direction (AF7 − AF8) removed so glancing left or
     right does not look like a blink.
3. **Detectors** (`detectors.py`):
   - Muscle burst: envelope above threshold for ≥ 120 ms. Then, 0.3 s after the burst
     started, if the forehead signal has moved more than the learned threshold (in the learned
     direction) → `EYEBROW_RAISE`, otherwise → `JAW_CLENCH`. An unclear burst is therefore
     always `STOP`. Before a profile has eyebrow calibration, every burst is `STOP` and is
     reported immediately; with eyebrow detection on, `STOP` comes 0.3 s after the clench
     starts (≈ 4.5 cm at 0.15 m/s).
   - `DOUBLE_BLINK`: two B peaks (40–600 ms wide) between 0.12 s and the learned max gap
     apart; a single blink just expires.
4. **Safety rules** - blink candidates overlapping a muscle burst or a head movement
   (gyroscope) are discarded; after a data gap ≥ 100 ms, blink events are muted for 0.6 s.
   Muscle events (`STOP`) always pass.

Why not a neural network: calibration gives ~24 labelled trials per user, the signals are
tens to hundreds of µV (far larger than EEG), and a rule-based detector can explain every
decision and runs with ~0.1–0.3 s latency. Every calibration is saved, so the same data can
train and benchmark an ML model later.

## Calibration

```
15 s rest -> JAW CLENCH x8 -> DOUBLE BLINK x8 -> EYEBROW RAISE x8          (~1.9 min)
screen: "[Block 2/4] JAW CLENCH"  "JAW CLENCH 3/8 >>> NOW <<<"  ...  "relax jaw"
```

What calibration learns per user: the blink pattern, the muscle-burst and blink thresholds,
the forehead-shift threshold and direction for jaw vs eyebrow, and the double-blink timing. If eyebrow raises and jaw clenches look
the same on someone's headset, eyebrow raise is switched off (every burst = `STOP`) and the
report says so. Natural blinks and small movements are expected: the rest data with natural blinks
teaches the threshold to ignore single blinks; trials spoiled by electrode shifts, Bluetooth
gaps or head movement are rejected and reported; a gesture that still comes out weak
(< 75 % replay accuracy) can be re-recorded on its own. The report shows per-gesture
accuracy, a confusion matrix and false triggers per minute at rest.

## Results

Real data: one user, three sessions on two days and two laptops. Each session was used once
for calibration and tested on itself and on the two others.

| | Jaw clench | Double blink |
|---|---|---|
| Same session (calibrated that session) | **100 %** (3/3 sessions) | 60–100 % (mean 87 %) |
| Other session (no recalibration) | **100 %** (6/6 pairs) | 0–100 % (mean 70 %) |
| False triggers at rest | none | 0–24 per minute, from natural blinks in quick pairs |

**Eyebrow raise** (one session, 8 jaw clenches + 8 eyebrow raises): the first version told
them apart by the forehead/ear EMG ratio and confused some jaw clenches with eyebrow raises
(gap between the groups 0.03 in log ratio). Using the forehead shift instead: threshold
334 µV with jaw at 10–40 µV and eyebrow at 630–955 µV; replay 16/16 and leave-one-out 16/16
correct. More sessions are needed for a cross-session number.

Jaw clench is reliable across sessions. Double blink depends on headset fit and on how
someone blinks naturally - recalibrate each session, and if `GO` fires by accident, use
firmer, more deliberate double blinks.

**Simulator** (`python -m muse_robot sim-test`, signal model based on the real recordings,
with natural blinks, NaN dropouts and frequent head movements): 95–100 % of 80 gestures over
three seeds, every jaw clench and eyebrow raise correct, 0–2 false commands in 6.8 min. The simulator is a regression test, not a
substitute for real data.

## Command line

```bash
python -m muse_robot check                                   # stream quality, NaN, contact
python -m muse_robot calibrate --profile profiles/me.json    # --trials 5 for a quicker run
python -m muse_robot run --profile profiles/me.json --output robot:192.168.1.11
python -m muse_robot run --profile profiles/me.json --output serial:COM5:115200
python -m muse_robot robot-test 192.168.1.11                 # drive the robot from the keyboard
python -m muse_robot fit recordings/calib_XXXX.npz --profile profiles/me.json   # re-fit
python -m muse_robot sim-test                                # no hardware needed
python -m pytest tests                                       # tests
```

`--source brainflow` connects to the Muse directly over Bluetooth instead of BlueMuse.
`--output` can be given several times (`print`, `robot:IP[:PORT]`, `udp:HOST:PORT`,
`serial:PORT[:BAUD]`). Gesture → command mapping: `DEFAULT_COMMAND_MAP` in `muse_robot/config.py`.
Recordings made with older versions (which also had look-left/right trials) can still be
re-fitted with `fit`.

## Project structure

```
MuseRobot.bat              Windows menu (sets up Python on first run)
muse_robot/
  config.py                constants, command map, per-user Profile (JSON)
  sources.py               BlueMuse/muselsl (LSL) + gyroscope, BrainFlow, simulator
  dsp.py                   NaN handling, filters, EMG envelopes (all / ear / forehead), blink signal
  detectors.py             muscle-burst (jaw vs eyebrow) and double-blink detectors, CommandEngine
  calibrate.py             cue protocol, trial rejection, fitting, replay evaluation
  outputs.py               screen, UDP to the robot (heartbeat + acks), serial
  sounds.py                optional beep when a command is detected
  cli.py                   command line
robot/
  eeg_cmd_bridge.py        runs ON THE ROBOT: UDP commands -> ROS 2 /cmd_vel, watchdog
  start_bridge.sh          starts the bridge on the robot
  README_robot.md          robot setup guide
tests/test_pipeline.py     pytest suite (filters, NaN, motion gating, STOP, bridge, simulator)
lsl_api.cfg                keeps the LSL library quiet
```

`profiles/` (calibration results) and `recordings/` (raw EEG of calibration sessions) are
created locally and are **not** committed: they are personal biosignal data.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No EEG stream found` | BlueMuse must be streaming (Start Streaming) |
| A channel is not `OK` in Check signal | reseat the headband; forehead pads flat, hair away from ear sensors |
| NaN values | short gaps are filled automatically; many gaps → move the laptop closer, avoid USB 3 ports next to the Bluetooth adapter |
| Double blinks missed | recalibrate; blink twice firmly, ~0.3 s apart |
| Eyebrow raise detected as STOP | raise higher with the teeth apart; forehead pads flat; redo the eyebrow block |
| Jaw clench detected as eyebrow | recalibrate; keep the eyebrows still while clenching |
| Random `GO` | natural quick double blinks; blink relaxed, use a firmer deliberate double blink |
| Robot `NO REPLY` | robot IP, same Wi-Fi, bridge running, allow Python in Windows Firewall (private networks) |

## Next steps

- Focus vs. relax (alpha/beta band power) as a continuous speed control.
- A trained classifier (e.g. EEGNet or gradient boosting) on the saved calibration recordings,
  benchmarked against the rule-based detectors.
- A graphical UI for calibration and live signal display.

## References

- BlueMuse - Muse streaming over LSL on Windows: <https://github.com/kowalej/BlueMuse>
- Lab Streaming Layer / pylsl: <https://github.com/sccn/labstreaminglayer>
- muse-lsl: <https://github.com/alexandrebarachant/muse-lsl>
- EEG-ExPy (NeuroTechX) - Muse validation experiments: <https://neurotechx.github.io/EEG-ExPy/>
- Krigolson et al. (2017), *Choosing MUSE: Validation of a Low-Cost, Portable EEG System for ERP
  Research*, Frontiers in Neuroscience: <https://www.frontiersin.org/articles/10.3389/fnins.2017.00109>
- Yahboom ROSMASTER A1: <https://www.yahboom.net/>
