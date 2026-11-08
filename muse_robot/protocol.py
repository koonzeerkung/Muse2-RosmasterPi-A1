"""The packet the EEG side sends to the robot side (protocol version 1).

One UDP datagram = one JSON object (UTF-8). Two message types:

event      sent once for every detected gesture
heartbeat  sent every 0.5 s, so the receiver knows the EEG side is alive

    {
      "v": 1,                          protocol version
      "type": "event",                 "event" | "heartbeat"
      "seq": 42,                       +1 for every message (spot lost / reordered packets)
      "time": 1762610000.123,          sender clock, Unix seconds
      "source": "muse2",
      "event": {                       only in "event" messages
        "gesture": "JAW_CLENCH",       JAW_CLENCH | DOUBLE_BLINK | EYEBROW_RAISE
        "command": "TOGGLE_RUN",       TOGGLE_RUN | TOGGLE_MODE | TOGGLE_MUSIC
        "t": 123.45,                   seconds since the EEG stream started
        "info": {...}                  detector details (for logging only)
      },
      "state": {                       state AFTER this event (in every message)
        "run": true,                   robot running (true) / stopped (false)
        "mode": "robot",               "robot" | "camera": what the gaze controls
        "music": false                 music playing
      }
    }

The receiver can either react to the toggle in "event", or simply apply "state"
(which is also in every heartbeat, so a lost packet is corrected within 0.5 s).
Recommended on the robot side: if no message arrives for ~1.5 s, stop the robot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import DEFAULT_COMMAND_MAP

VERSION = 1
MODES = ("robot", "camera")

# gesture -> command: edit DEFAULT_COMMAND_MAP in config.py if the team changes the mapping
COMMANDS = DEFAULT_COMMAND_MAP


@dataclass
class ToggleState:
    run: bool = False          # start stopped
    mode: str = "robot"        # what the gaze controls first
    music: bool = False

    def apply(self, command: str) -> None:
        if command == "TOGGLE_RUN":
            self.run = not self.run
        elif command == "TOGGLE_MODE":
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        elif command == "TOGGLE_MUSIC":
            self.music = not self.music

    def as_dict(self) -> dict:
        return {"run": self.run, "mode": self.mode, "music": self.music}

    def text(self) -> str:
        return f"run={'ON ' if self.run else 'off'} mode={self.mode:6s} music={'ON ' if self.music else 'off'}"


class PacketBuilder:
    """Turns gestures into protocol messages and keeps the toggle state + sequence number."""

    def __init__(self, state: ToggleState | None = None, source: str = "muse2"):
        self.state = state or ToggleState()
        self.source = source
        self.seq = 0

    def _base(self, kind: str) -> dict:
        self.seq += 1
        return {"v": VERSION, "type": kind, "seq": self.seq, "time": round(time.time(), 3),
                "source": self.source}

    def event(self, gesture: str, t: float = 0.0, info: dict | None = None) -> dict | None:
        command = COMMANDS.get(gesture)
        if command is None:
            return None
        self.state.apply(command)
        msg = self._base("event")
        msg["event"] = {"gesture": gesture, "command": command, "t": round(float(t), 3), "info": info or {}}
        msg["state"] = self.state.as_dict()
        return msg

    def heartbeat(self) -> dict:
        msg = self._base("heartbeat")
        msg["state"] = self.state.as_dict()
        return msg
