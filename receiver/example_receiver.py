#!/usr/bin/env python3
"""Example receiver for the Muse 2 EEG packets - a starting point for the robot side.

Standard library only (Python 3.8+), runs on Windows, Linux or the robot's Raspberry Pi.
Fill in the three on_... functions with the real robot / camera / music code.

    python example_receiver.py                 # listen on UDP port 9000
    python example_receiver.py --port 9100
    python example_receiver.py --use-events    # react to each toggle instead of the state

Packet format: see PROTOCOL.md next to this file.
"""
from __future__ import annotations

import argparse
import json
import socket
import time

LINK_TIMEOUT_S = 1.5     # no message for this long -> treat the EEG side as gone


# ---------------------------------------------------------------- put the real robot code here
def on_run(run: bool) -> None:
    print(f"    -> robot {'RUN' if run else 'STOP'}")


def on_mode(mode: str) -> None:
    print(f"    -> gaze now controls the {mode.upper()}")


def on_music(music: bool) -> None:
    print(f"    -> music {'PLAY' if music else 'PAUSE'}")


def on_link_lost() -> None:
    """EEG laptop silent (closed, asleep, Wi-Fi lost): make the robot safe."""
    print("    -> link lost: robot STOP")
    on_run(False)


# ---------------------------------------------------------------- receiving logic
class Receiver:
    def __init__(self, use_events: bool = False):
        self.use_events = use_events
        self.state = {"run": False, "mode": "robot", "music": False}
        self.last_msg = None
        self.last_seq = None
        self.link = False

    def apply_state(self, new: dict) -> None:
        """Act only on what changed (state is repeated in every heartbeat)."""
        if "run" in new and new["run"] != self.state["run"]:
            self.state["run"] = bool(new["run"])
            on_run(self.state["run"])
        if "mode" in new and new["mode"] != self.state["mode"]:
            self.state["mode"] = new["mode"]
            on_mode(self.state["mode"])
        if "music" in new and new["music"] != self.state["music"]:
            self.state["music"] = bool(new["music"])
            on_music(self.state["music"])

    def apply_toggle(self, command: str) -> None:
        if command == "TOGGLE_RUN":
            self.apply_state({"run": not self.state["run"]})
        elif command == "TOGGLE_MODE":
            self.apply_state({"mode": "camera" if self.state["mode"] == "robot" else "robot"})
        elif command == "TOGGLE_MUSIC":
            self.apply_state({"music": not self.state["music"]})

    def handle(self, msg: dict, now: float) -> None:
        if msg.get("v") != 1:
            print(f"  unknown protocol version: {msg.get('v')}")
            return
        seq = msg.get("seq")
        if self.last_seq is not None and isinstance(seq, int):
            if seq <= self.last_seq and seq > self.last_seq - 1000:
                return                                     # old / duplicated packet
            if seq != self.last_seq + 1:
                print(f"  (lost {seq - self.last_seq - 1} packet(s))")
        self.last_seq = seq
        self.last_msg = now
        if not self.link:
            self.link = True
            print("  link up")
            if not self.use_events:                       # sync to the sender right away
                self.apply_state(msg.get("state", {}))
        if msg.get("type") == "event":
            e = msg["event"]
            print(f"  #{seq} {e['gesture']} -> {e['command']}")
            if self.use_events:
                self.apply_toggle(e["command"])
            else:
                self.apply_state(msg.get("state", {}))
        elif msg.get("type") == "heartbeat" and not self.use_events:
            self.apply_state(msg.get("state", {}))        # corrects a lost event packet

    def check_link(self, now: float) -> None:
        if self.link and now - self.last_msg > LINK_TIMEOUT_S:
            self.link = False
            self.last_seq = None                          # sender may restart from seq 1
            print(f"  no message for {LINK_TIMEOUT_S} s")
            on_link_lost()
            self.state["run"] = False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--use-events", action="store_true",
                    help="toggle on each event instead of following the 'state' field")
    ap.add_argument("--no-reply", action="store_true", help="don't send the small reply packet")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(0.2)
    rx = Receiver(a.use_events)
    print(f"listening for Muse 2 packets on UDP port {a.port} - Ctrl+C to stop")
    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                rx.check_link(time.monotonic())
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            rx.handle(msg, time.monotonic())
            if not a.no_reply:                            # lets the EEG screen show "receiver replies"
                try:
                    sock.sendto(json.dumps({"ack": msg.get("seq"), "state": rx.state}).encode(), addr)
                except OSError:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        on_run(False)
        sock.close()


if __name__ == "__main__":
    main()
