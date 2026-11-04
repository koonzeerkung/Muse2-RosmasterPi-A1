# Muse 2 → Yahboom RosmasterPi A1

Drive a **Yahboom RosmasterPi A1** robot car with a **Muse 2** EEG headband. Four deliberate
face/eye gestures are detected in real time from the headband's 4 channels and sent over Wi-Fi
to the robot, which drives through ROS 2.

| Gesture | Detected from | Command | Robot (Ackermann car) |
|---|---|---|---|
| **Jaw clench** | muscle activity (EMG, 30–110 Hz) on all 4 sensors | `STOP` | stops at once (always wins) |
| **Double blink** | two blink peaks ~0.2–0.5 s apart | `GO` | drives forward |
| **Look left** (and back) | eye-movement spike, then the opposite spike | `LEFT` | steers left for 1.5 s |
| **Look right** (and back) | same, other direction | `RIGHT` | steers right for 1.5 s |

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
   1. Check signal   (NaN, contact quality, 20 s)
   2. Calibrate      (about 3 minutes)
   3. Run            - show commands on screen only
   4. Run            - send to a SERIAL device (COM port)
   5. Run            - drive the YAHBOOM ROBOT over Wi-Fi
   6. Change profile name
   7. Test without headset (simulator)
   8. Test robot connection (keyboard, no headset)
   ```

3. **1 - Check signal**: every channel should say `OK`. If not, push the headband down so the
   forehead pads sit flat, move hair away from the ear sensors and check again.
4. **2 - Calibrate** (~3 min, prompts on screen, cue sounds for the look trials).
5. **3 - Run** to see the commands on screen, then **5 - Run** to drive the robot
   (after the [robot setup](#robot-setup)). While running, `Ctrl+C` then `N` returns to the menu.

### Doing the gestures

- **Jaw clench** - bite down firmly for about a second. Most reliable, so it is `STOP`.
- **Double blink** - two quick, firm blinks.
- **Look left / right** - move **only your eyes** far to the side, hold at least 0.5 s, come back
  to the screen. Keep your head still. The next look is accepted about 1 s later.

## Robot setup

Full guide: **[robot/README_robot.md](robot/README_robot.md)**. In short:

1. Robot and laptop on the same Wi-Fi; note the robot's IP address.
2. Copy the `robot/` folder to the robot, e.g.
   `scp -r robot pi@ROBOT_IP:~/eeg_bridge`.
3. On the robot: start the chassis driver as in Yahboom's A1 tutorial (`yahboomcar_bringup`,
   the same one used for keyboard/joystick control), check `ros2 topic info /cmd_vel`, then
   `bash ~/eeg_bridge/start_bridge.sh`.
4. On the laptop: **8 - Test robot connection** (drive with `g`/`a`/`d`/`s`, wheels off the ground
   the first time), then **5 - Run**.

**Safety:** the laptop sends a heartbeat twice a second; if the robot hears nothing for 1.5 s
(program closed, laptop asleep, Wi-Fi lost) it stops by itself and waits for a new `GO`.
Closing the laptop program sends `STOP`. `STOP` is never blocked by any filter.
Speed (default 0.15 m/s), steering, turn time and timeouts are options of `start_bridge.sh`.

## How it works

### What the Muse 2 actually records

Measured on real recordings with this project:

- Muse 2's reference electrode (Fpz) sits on the forehead **right above the eyes**. A blink is
  therefore seen mostly as a large *negative* dip (≈ −250 µV) on the **ear** sensors TP9/TP10,
  and only as a small wiggle on the forehead sensors AF7/AF8.
- An eye movement to the side gives a short **spike** (~0.3 s) - AF7 and AF8 in opposite
  directions - that fades even while the eyes stay to the side. Looking back gives the
  opposite spike.
- Clenching the jaw produces broadband muscle activity (EMG) far above the EEG band.

### Pipeline (`muse_robot/`)

1. **Clean** - NaN samples from Bluetooth dropouts are filled with the last good value (IIR
   filters would otherwise turn one NaN into NaN forever); 50/100 Hz mains notch.
2. **Features** - all filters are causal and keep state between chunks, so offline replay of a
   recording gives exactly the same events as live:
   - **EMG envelope**: 30–110 Hz band, RMS over 100 ms, averaged over the 4 channels.
   - **Blink signal B** and **gaze signal H**: the 0.5–10 Hz signal projected on two
     4-channel patterns learned in calibration (a blink, and a *left* look), each made
     orthogonal to the other so a blink barely moves H and a look barely moves B.
3. **Detectors** (`detectors.py`):
   - `JAW_CLENCH`: envelope above threshold for ≥ 120 ms.
   - `DOUBLE_BLINK`: two B peaks (40–600 ms wide) between 0.12 s and the learned max gap apart;
     a single blink just expires.
   - `LOOK_LEFT/RIGHT`: an H spike, then an opposite spike 0.45 s to ~2.4 s later; the sign of
     the first spike is the direction. A double blink leaks into H as spikes only ~0.3 s apart,
     so it cannot form a look; filter rebounds right after a spike and leftovers for 1 s after
     a look are ignored.
4. **Safety rules** - blink/look candidates overlapping a jaw clench or a head movement
   (gyroscope) are discarded; after a data gap ≥ 100 ms, blink/look events are muted for 0.6 s.
   `STOP` always passes.

Why not a neural network for these four commands: calibration gives ~32 labelled trials per
user, the signals are tens to hundreds of µV (far larger than EEG), and a rule-based detector
can explain every decision and runs with ~0.1–0.3 s latency. Every calibration is saved, so the
same data can train and benchmark an ML model later.

## Calibration

```
15 s rest -> JAW x8 -> DOUBLE BLINK x8 -> LOOK LEFT x8 -> LOOK RIGHT x8        (~2.6 min)
screen: "[Block 4/5] LOOK LEFT"  "LOOK LEFT 3/8 >>> NOW <<<"
```

| Trial | Cue |
|---|---|
| Jaw clench | screen only: `>>> NOW <<<` → clench, `relax jaw` 1 s later → release |
| Double blink | screen only: `>>> NOW <<<` → blink twice |
| Look left / right | `signal.wav` from the speakers → eyes to the block's side and hold; **1 s after the sound ends**, a low beep → eyes back to the screen |

The look trials need a sound because the eyes are off the screen; the fixed return beep gives
every trial the same hold time, so the data are consistent.

What calibration learns per user: the blink and look patterns, all thresholds, the double-blink
timing and the look out-and-back timing. Natural blinks and small movements are expected:
trials spoiled by electrode shifts, Bluetooth gaps or head movement are rejected and reported;
gestures that still come out weak (< 75 % replay accuracy) can be re-recorded on their own.
The report shows per-gesture accuracy, a confusion matrix and false triggers per minute at rest.

## Results

**Real data** (one user, two separate sessions on the same day; calibrate on session 2, test on
session 1 - a different session than the one used for fitting):

| Gesture | Accuracy |
|---|---|
| Jaw clench | 100 % |
| Double blink | 100 % |
| Look left | 75 % (the miss was a smaller-than-usual eye movement) |
| Look right | 100 % |
| **Overall** | **94.7 %** |

False triggers come mainly from **natural blinks in quick pairs**, which look exactly like a
double blink (≈ 1 per 15 s of rest in one session). This is the gesture to watch; the
calibration report warns when it happens.

**Simulator** (`python -m muse_robot sim-test`, signal model based on the real recordings, with
natural blinks, NaN dropouts and frequent head movements): 84–98 % of 80 gestures over three
seeds, 0–1 false commands in 6.8 min. The simulator is a regression test, not a substitute for
real data.

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

## Project structure

```
MuseRobot.bat              Windows menu (sets up Python on first run)
muse_robot/
  config.py                constants, command map, per-user Profile (JSON)
  sources.py               BlueMuse/muselsl (LSL) + gyroscope, BrainFlow, simulator
  dsp.py                   NaN handling, filters, EMG / blink / gaze signals
  detectors.py             jaw, double-blink and look detectors, CommandEngine
  calibrate.py             cue protocol, trial rejection, fitting, replay evaluation
  outputs.py               screen, UDP to the robot (heartbeat + acks), serial
  sounds.py                calibration cue sounds (uses signal.wav if present)
  cli.py                   command line
robot/
  eeg_cmd_bridge.py        runs ON THE ROBOT: UDP commands -> ROS 2 /cmd_vel, watchdog
  start_bridge.sh          starts the bridge on the robot
  README_robot.md          robot setup guide
tests/test_pipeline.py     pytest suite (filters, NaN, motion gating, STOP, bridge, simulator)
signal.wav                 go cue for the look trials
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
| Blinks / looks missed | recalibrate; make looks bigger and hold ≥ 0.5 s; firm double blinks |
| Random `GO` | natural quick double blinks; blink relaxed, use a firmer deliberate double blink |
| Robot `NO REPLY` | robot IP, same Wi-Fi, bridge running, allow Python in Windows Firewall (private networks) |
| Robot turns the wrong way | `bash start_bridge.sh --steer -0.6` |

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
