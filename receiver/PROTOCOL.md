# Muse 2 → robot packet protocol (version 1)

The EEG laptop (`MuseRobot.bat` → **4. Run – send packets**) detects three gestures and sends
one small JSON message per gesture over the network. This file is everything the robot side
needs to receive them.

## Gestures and commands

| Gesture | Command | Meaning | State field |
|---|---|---|---|
| Jaw clench | `TOGGLE_RUN` | robot run ↔ stop | `run`: `true` = running |
| Double blink | `TOGGLE_MODE` | gaze controls the robot movement ↔ the camera | `mode`: `"robot"` / `"camera"` |
| Eyebrow raise | `TOGGLE_MUSIC` | music on ↔ off | `music`: `true` = playing |

Start state (before any gesture): `run=false`, `mode="robot"` (option `--start-mode camera`),
`music=false`.

## Transport

- **UDP**, one JSON object per datagram, UTF-8, default port **9000**
  (`udp:HOST:PORT` on the EEG side; the HOST is the robot-side machine).
- Optional: **serial**, one JSON object per line (`serial:COM5:115200`).
- Optional reply: any datagram the receiver sends back to the sender's address is taken as
  "receiver alive" and shown on the EEG screen. Its content is not read.

## Messages

**event** - once for every detected gesture:

```json
{
  "v": 1,
  "type": "event",
  "seq": 42,
  "time": 1762610000.123,
  "source": "muse2",
  "event": {
    "gesture": "JAW_CLENCH",
    "command": "TOGGLE_RUN",
    "t": 123.45,
    "info": {"env": 61.2, "ratio": -0.5, "shift": 40.1}
  },
  "state": {"run": true, "mode": "robot", "music": false}
}
```

**heartbeat** - every **0.5 s**, same fields without `event`:

```json
{"v": 1, "type": "heartbeat", "seq": 43, "time": 1762610000.623, "source": "muse2",
 "state": {"run": true, "mode": "robot", "music": false}}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | int | protocol version, `1` |
| `type` | string | `"event"` or `"heartbeat"` |
| `seq` | int | +1 for every message from this sender (starts at 1 when the EEG program starts) |
| `time` | float | sender clock, Unix seconds |
| `source` | string | `"muse2"` |
| `event.gesture` | string | `JAW_CLENCH`, `DOUBLE_BLINK`, `EYEBROW_RAISE` |
| `event.command` | string | `TOGGLE_RUN`, `TOGGLE_MODE`, `TOGGLE_MUSIC` |
| `event.t` | float | seconds since the EEG stream started |
| `event.info` | object | detector details, for logging only - may change |
| `state` | object | `run` (bool), `mode` (`"robot"`/`"camera"`), `music` (bool) **after** this message |

## How to use it on the robot side

Two ways, pick one:

1. **Follow `state` (recommended).** Apply `state` from every message and act only when a
   field changes. Because `state` is repeated in every heartbeat, a lost UDP packet is
   corrected within 0.5 s and both sides can never drift apart.
2. **React to `event.command`.** Flip your own state on each toggle. Simpler, but a lost
   packet leaves the two sides out of step.

**Safety:** if no message arrives for **1.5 s** (EEG laptop closed or asleep, Wi-Fi lost),
stop the robot. When messages come back, continue from their `state` (with option 1 the robot
only runs again if `state.run` is true).

Ignore messages with `seq` lower than the last one (late/duplicated UDP), except when the link
was lost - the EEG program may have restarted and begun again at `seq` 1.

## What is NOT sent

- Muscle bursts the program can't classify (forehead shift close to the jaw/eyebrow boundary)
  are shown on the EEG screen as "ignored" and **not sent** - every command is a toggle, so a
  wrong guess would flip the wrong thing.
- Blinks/look movements that are not a deliberate double blink.

## Testing without the robot

On the EEG laptop (`MuseRobot.bat`):

- **7. Send test packets (keyboard)** - `j` / `b` / `e` send the three toggles to an IP and port
  (no headset needed).
- **8. Listen for packets** - shows what arrives on a port (a stand-in for the robot side).

On the robot-side machine: `python example_receiver.py` (standard library only) - prints the
packets and has `on_run`, `on_mode`, `on_music`, `on_link_lost` functions to fill in.
