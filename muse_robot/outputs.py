"""Where commands go: console, UDP/JSON (Yahboom robot bridge), or a serial port (Arduino, ESP32, ...)."""
from __future__ import annotations

import json
import socket

from .detectors import Event


class PrintSink:
    def send(self, ev: Event, command: str) -> None:
        print(f"\n[{ev.t:8.2f}s] {ev.gesture:13s} -> {command}   {ev.info}", flush=True)

    def close(self):
        pass


class UDPSink:
    """One JSON datagram per command, e.g. to robot/eeg_cmd_bridge.py on the Yahboom robot.

    Also sends a heartbeat (call heartbeat() every ~0.5 s) so the robot can stop by
    itself if the laptop or Wi-Fi goes away, and reads the robot's acks to show
    whether it is connected.
    """

    def __init__(self, host: str, port: int):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.last_ack = None           # time.monotonic() of the last reply from the robot
        self.robot = {}                # last state reported by the robot

    def _send(self, msg: dict) -> None:
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except OSError:
            pass                       # Wi-Fi down: the robot's watchdog stops it
        self.poll()

    def send(self, ev: Event | None, command: str) -> None:
        msg = {"type": "command", "command": command}
        if ev is not None:
            msg.update({"gesture": ev.gesture, "t": round(ev.t, 3), **ev.info})
        self._send(msg)

    def heartbeat(self) -> None:
        self._send({"type": "heartbeat"})

    def poll(self) -> None:
        import time
        while True:
            try:
                data, _ = self.sock.recvfrom(2048)
            except (BlockingIOError, OSError):
                return
            try:
                self.robot = json.loads(data.decode())
                self.last_ack = time.monotonic()
            except ValueError:
                pass

    def connected(self, within: float = 1.5) -> bool:
        import time
        return self.last_ack is not None and time.monotonic() - self.last_ack < within

    def status_text(self) -> str:
        if not self.connected():
            return f"robot {self.addr[0]}: NO REPLY"
        r = self.robot
        return f"robot: {r.get('state', '?')}{' ' + r['turn'] if r.get('turn') else ''} v={r.get('vx', 0)}"

    def close(self):
        try:
            self.send(None, "STOP")    # leave the robot stopped
        finally:
            self.sock.close()


class SerialSink:
    """Writes the command word + newline, e.g. b"LEFT\\n". Needs `pip install pyserial`."""

    def __init__(self, port: str, baud: int = 115200):
        import serial

        self.ser = serial.Serial(port, int(baud), timeout=0)

    def send(self, ev: Event, command: str) -> None:
        self.ser.write((command + "\n").encode())

    def close(self):
        self.ser.close()


def make_sink(spec: str):
    """'print' | 'udp:HOST:PORT' | 'robot:IP[:PORT]' | 'serial:PORT[:BAUD]'  (PORT like COM5)"""
    kind, _, rest = spec.partition(":")
    if kind == "print":
        return PrintSink()
    if kind in ("udp", "robot"):
        host, _, port = rest.rpartition(":")
        if kind == "robot" and not port.isdigit():      # robot:IP  -> default port 9000
            host, port = rest, "9000"
        return UDPSink(host or "127.0.0.1", int(port))
    if kind == "serial":
        port, _, baud = rest.rpartition(":")
        if not port or not baud.isdigit():          # no baud given
            port, baud = rest, "115200"
        return SerialSink(port, int(baud))
    raise ValueError(f"unknown output '{spec}'")
