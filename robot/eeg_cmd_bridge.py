#!/usr/bin/env python3
"""EEG command bridge - runs ON THE ROBOT (Yahboom RosmasterPi A1, ROS 2 Humble).

The EEG laptop (Windows, BlueMuse + muse_robot) sends small UDP/JSON packets over
Wi-Fi; this node turns them into geometry_msgs/Twist on /cmd_vel for the A1's
Ackermann chassis driver (yahboomcar_bringup). No ROS is needed on the laptop.

    laptop  --UDP:9000-->  eeg_cmd_bridge  --/cmd_vel-->  chassis driver  --> motors
            <--ack/state--

Commands (from the EEG gestures):
    GO     (double blink)  drive forward at --speed, wheels straight
    STOP   (jaw clench)    stop immediately (always wins)
    LEFT   (look left)     steer left for --turn-time seconds, then straight again
    RIGHT  (look right)    steer right for --turn-time seconds
    An Ackermann car can't turn in place, so LEFT/RIGHT while stopped drives a short
    arc (--turn-time) at --speed and then stops again.

Safety
    * link watchdog: the laptop sends a heartbeat every 0.5 s; if nothing arrives for
      --link-timeout seconds (laptop closed, Wi-Fi lost), the robot stops and waits
      for a new GO.
    * optional --max-drive: stop automatically after N seconds of driving without
      a new command (0 = off).
    * speeds are clamped to --max-speed.

Run (on the robot, in the ROS 2 environment / Yahboom docker with --net=host):
    1) start the chassis driver as in Yahboom's tutorial (yahboomcar_bringup)
    2) python3 eeg_cmd_bridge.py                     # defaults: port 9000, 0.15 m/s
       python3 eeg_cmd_bridge.py --speed 0.2 --steer 0.6 --turn-time 1.5
       python3 eeg_cmd_bridge.py --dry-run           # no ROS: just print velocities
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time


class BridgeCore:
    """Pure command/state logic (no ROS), so it can be tested anywhere."""

    def __init__(self, speed=0.15, steer=0.6, turn_time=1.5, link_timeout=1.5, max_drive=0.0, max_speed=0.4):
        self.speed = min(abs(speed), max_speed)
        self.steer = steer             # negative value flips left/right if the car turns the wrong way
        self.turn_time = turn_time
        self.link_timeout = link_timeout
        self.max_drive = max_drive
        self.state = "STOP"            # STOP | DRIVE
        self.turn_dir = 0              # +1 left, -1 right, 0 straight
        self.turn_until = 0.0
        self.last_packet = None        # time of last packet from the laptop
        self.last_cmd_t = 0.0
        self.last_cmd = None
        self.link_ok = False
        self.lock = threading.Lock()

    def on_packet(self, now: float) -> None:
        with self.lock:
            self.last_packet = now
            self.link_ok = True

    def on_command(self, cmd: str, now: float) -> None:
        cmd = cmd.strip().upper()
        with self.lock:
            self.last_packet = now
            self.link_ok = True
            self.last_cmd, self.last_cmd_t = cmd, now
            if cmd == "STOP":
                self.state, self.turn_dir = "STOP", 0
            elif cmd == "GO":
                self.state, self.turn_dir = "DRIVE", 0
            elif cmd in ("LEFT", "RIGHT"):
                self.turn_dir = 1 if cmd == "LEFT" else -1
                self.turn_until = now + self.turn_time

    def velocity(self, now: float) -> tuple[float, float]:
        """(linear.x m/s, angular.z) to publish now. Applies timeouts."""
        with self.lock:
            if self.last_packet is not None and now - self.last_packet > self.link_timeout:
                if self.link_ok:
                    self.link_ok = False
                    self.state, self.turn_dir = "STOP", 0
            if self.turn_dir and now >= self.turn_until:
                self.turn_dir = 0
            if self.state == "DRIVE" and self.max_drive and now - self.last_cmd_t > self.max_drive:
                self.state = "STOP"
            moving = self.state == "DRIVE" or self.turn_dir != 0
            vx = self.speed if moving and self.link_ok else 0.0
            wz = self.steer * self.turn_dir if vx else 0.0
            return vx, wz

    def status(self, now: float) -> dict:
        vx, wz = self.velocity(now)
        with self.lock:
            return {"state": self.state, "turn": {1: "LEFT", -1: "RIGHT", 0: ""}[self.turn_dir],
                    "vx": round(vx, 3), "wz": round(wz, 3), "link": self.link_ok, "last_cmd": self.last_cmd}


def udp_loop(core: BridgeCore, port: int, stop: threading.Event, log=print) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.1)
    log(f"[bridge] listening for EEG commands on UDP port {port}")
    peer = None
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        now = time.monotonic()
        try:
            msg = json.loads(data.decode())
        except (ValueError, UnicodeDecodeError):
            msg = {"command": data.decode(errors="ignore")}       # plain text "GO\n" also works
        if addr != peer:
            log(f"[bridge] laptop connected from {addr[0]}")
            peer = addr
        if msg.get("type") == "heartbeat":
            core.on_packet(now)
        elif msg.get("command"):
            core.on_command(str(msg["command"]), now)
            log(f"[bridge] {msg.get('gesture', '')} -> {msg['command']}")
        reply = {"type": "ack", **core.status(now)}
        try:
            sock.sendto(json.dumps(reply).encode(), addr)
        except OSError:
            pass
    sock.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--topic", default="/cmd_vel")
    ap.add_argument("--speed", type=float, default=0.15, help="forward speed, m/s (start slow!)")
    ap.add_argument("--steer", type=float, default=0.6,
                    help="angular.z sent for LEFT (RIGHT = minus this). Use a negative value if turns are reversed")
    ap.add_argument("--turn-time", type=float, default=1.5, help="seconds a LEFT/RIGHT steers")
    ap.add_argument("--link-timeout", type=float, default=1.5, help="stop if the laptop is silent this long")
    ap.add_argument("--max-drive", type=float, default=0.0, help="auto-stop after N s without a command (0=off)")
    ap.add_argument("--max-speed", type=float, default=0.4)
    ap.add_argument("--rate", type=float, default=20.0, help="publish rate, Hz")
    ap.add_argument("--dry-run", action="store_true", help="no ROS: print the velocities instead")
    a = ap.parse_args()

    core = BridgeCore(a.speed, a.steer, a.turn_time, a.link_timeout, a.max_drive, a.max_speed)
    stop = threading.Event()
    threading.Thread(target=udp_loop, args=(core, a.port, stop), daemon=True).start()

    if a.dry_run:
        last = None
        try:
            while True:
                v = core.velocity(time.monotonic())
                if v != last:
                    print(f"[dry-run] linear.x={v[0]:.3f}  angular.z={v[1]:.3f}   {core.status(time.monotonic())}")
                    last = v
                time.sleep(1.0 / a.rate)
        except KeyboardInterrupt:
            stop.set()
        return

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    class EEGBridge(Node):
        def __init__(self):
            super().__init__("eeg_cmd_bridge")
            self.pub = self.create_publisher(Twist, a.topic, 10)
            self.zero_left = 0
            self.prev = (0.0, 0.0)
            self.create_timer(1.0 / a.rate, self.tick)
            self.get_logger().info(f"publishing {a.topic} at {a.rate:.0f} Hz, speed {core.speed} m/s, "
                                   f"steer {core.steer}, turn {a.turn_time}s, link timeout {a.link_timeout}s")

        def tick(self):
            vx, wz = core.velocity(time.monotonic())
            if (vx, wz) != self.prev:
                self.get_logger().info(f"cmd_vel linear.x={vx:.2f} angular.z={wz:.2f}  ({core.status(time.monotonic())['state']})")
            if vx or wz:
                self.zero_left = 10                  # after stopping, send zeros for 0.5 s then go quiet
            elif self.zero_left == 0:
                self.prev = (vx, wz)                 # idle: don't fight other controllers (joystick/app)
                return
            else:
                self.zero_left -= 1
            msg = Twist()
            msg.linear.x, msg.angular.z = float(vx), float(wz)
            self.pub.publish(msg)
            self.prev = (vx, wz)

        def stop_robot(self):
            for _ in range(5):
                self.pub.publish(Twist())
                time.sleep(0.02)

    rclpy.init()
    node = EEGBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
