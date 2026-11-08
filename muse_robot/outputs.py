"""Where the protocol messages go (see protocol.py for the format).

  print              readable line on screen for every event
  udp:HOST:PORT      one JSON datagram per message (events + heartbeats) - the robot side
  serial:PORT[:BAUD] one JSON line per message (events + heartbeats), e.g. COM5:115200
"""
from __future__ import annotations

import json
import socket
import time


class PrintSink:
    def send(self, msg: dict) -> None:
        if msg.get("type") != "event":
            return
        e = msg["event"]
        s = msg["state"]
        print(f"\n[{e['t']:8.2f}s] {e['gesture']:13s} -> {e['command']:12s}  "
              f"run={'ON ' if s['run'] else 'off'} mode={s['mode']:6s} music={'ON' if s['music'] else 'off'}"
              f"   (#{msg['seq']})", flush=True)

    def status_text(self) -> str:
        return ""

    def close(self):
        pass


class UDPSink:
    """Sends every message as one JSON datagram. Reads optional replies from the receiver
    (any JSON datagram sent back to us) only to show "receiver: connected" on screen."""

    def __init__(self, host: str, port: int):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.last_reply = None
        self.sent = 0

    def send(self, msg: dict) -> None:
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
            self.sent += 1
        except OSError:
            pass                       # network down: the receiver notices the missing heartbeats
        self._poll()

    def _poll(self) -> None:
        while True:
            try:
                self.sock.recvfrom(4096)
                self.last_reply = time.monotonic()
            except (BlockingIOError, OSError):
                return

    def status_text(self) -> str:
        replied = self.last_reply is not None and time.monotonic() - self.last_reply < 2.0
        return f"udp {self.addr[0]}:{self.addr[1]} sent={self.sent}" + (" (receiver replies)" if replied else "")

    def close(self):
        self.sock.close()


class SerialSink:
    """One JSON line per message. Needs `pip install pyserial`."""

    def __init__(self, port: str, baud: int = 115200):
        import serial

        self.ser = serial.Serial(port, int(baud), timeout=0)

    def send(self, msg: dict) -> None:
        self.ser.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    def status_text(self) -> str:
        return f"serial {self.ser.port}"

    def close(self):
        self.ser.close()


def make_sink(spec: str):
    """'print' | 'udp:HOST:PORT' | 'serial:PORT[:BAUD]'  (PORT like COM5 or /dev/ttyUSB0)"""
    kind, _, rest = spec.partition(":")
    if kind == "print":
        return PrintSink()
    if kind == "udp":
        host, _, port = rest.rpartition(":")
        if not port.isdigit():                          # udp:HOST -> default port 9000
            host, port = rest, "9000"
        return UDPSink(host or "127.0.0.1", int(port))
    if kind == "serial":
        port, _, baud = rest.rpartition(":")
        if not port or not baud.isdigit():
            port, baud = rest, "115200"
        return SerialSink(port, int(baud))
    raise ValueError(f"unknown output '{spec}' (use print, udp:HOST:PORT or serial:PORT[:BAUD])")
