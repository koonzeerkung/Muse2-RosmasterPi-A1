# Connecting the EEG laptop to the Yahboom RosmasterPi A1

```
Muse 2 --BLE--> laptop (Windows): BlueMuse + MuseRobot.bat
                   |
                   |  Wi-Fi, UDP port 9000  (commands + heartbeat 2x/s, robot replies with its state)
                   v
RosmasterPi A1 (Raspberry Pi 5, ROS 2 Humble): eeg_cmd_bridge.py --/cmd_vel--> chassis driver --> motors
```

No ROS is needed on the laptop. Only `eeg_cmd_bridge.py` runs on the robot; it publishes
`geometry_msgs/Twist` on `/cmd_vel`, the topic the A1 chassis driver (`yahboomcar_bringup`)
listens to - the same one Yahboom's keyboard / joystick control uses.

| EEG gesture | Command | What the A1 does |
|---|---|---|
| Double blink | GO | drive forward (`--speed`, default 0.15 m/s), wheels straight |
| Jaw clench | STOP | stop immediately - always wins |
| Eyebrow raise | BROW | nothing yet (logged) - choose a meaning and add it to the bridge |
| (keyboard test only) | LEFT / RIGHT | steer for `--turn-time` (1.5 s), then straight again |

The EEG side sends GO, STOP and BROW. The bridge still accepts LEFT/RIGHT (the A1 has
Ackermann steering, so a turn while stopped drives a short arc), which the keyboard test
uses and another program could send.

**Safety:** if the robot hears nothing from the laptop for 1.5 s (program closed, laptop
asleep, Wi-Fi lost) it stops and waits for a new GO. Quitting the laptop program sends STOP.
For the first test, lift the robot so the wheels are off the ground.

## One-time setup

1. **Same network.** Put the robot and the laptop on the same Wi-Fi (or connect the laptop to
   the robot's own hotspot if you use that mode). Find the robot's IP address: it is shown by
   `start_bridge.sh`, by `hostname -I` in a robot terminal, or in your router's device list.
2. **Copy this `robot` folder to the robot**, e.g. from a Command Prompt in the project folder
   on the laptop (Windows 10/11 has `scp` built in; user name/password as in Yahboom's manual):
   ```
   scp -r robot pi@ROBOT_IP:~/eeg_bridge
   ```
   A USB stick or VS Code Remote-SSH works too.
3. **Windows firewall:** the first time the laptop program runs, Windows may ask whether
   Python may use the network - allow it on *Private* networks. If "Test robot connection"
   says NO REPLY although the robot reacts, the firewall is blocking the replies.

## Every session

On the robot (SSH or a terminal on the robot):

1. Start the chassis driver exactly as in Yahboom's A1 tutorial for keyboard/joystick control
   (the `yahboomcar_bringup` launch). If your image runs ROS 2 inside Yahboom's Docker
   container, open a shell in that container for the next steps (it must use host networking,
   which Yahboom's start scripts normally do).
2. Check that something listens on `/cmd_vel`:
   ```
   ros2 topic info /cmd_vel        # "Subscription count: 1" or more
   ```
3. Start the bridge:
   ```
   cd ~/eeg_bridge && bash start_bridge.sh
   ```

On the laptop (`MuseRobot.bat`):

4. **8. Test robot connection** - enter the robot IP, then `g` (GO), `a`/`d` (steer LEFT/RIGHT),
   `s` (STOP), `q` (quit). You should see `CONNECTED` and the robot's state after each key.
5. **5. Run - drive the YAHBOOM ROBOT over Wi-Fi** - BlueMuse streaming, profile calibrated.
   The status line shows the signal and `robot: DRIVE/STOP ...`; `NO REPLY` means the link
   is down (the robot will have stopped by itself).

## Tuning (options for `start_bridge.sh` / `eeg_cmd_bridge.py`)

```
bash start_bridge.sh --speed 0.2          # forward speed, m/s (max 0.4 unless --max-speed)
bash start_bridge.sh --steer 0.8          # how hard it steers (angular.z)
bash start_bridge.sh --steer -0.6         # LEFT and RIGHT are reversed on your car? flip the sign
bash start_bridge.sh --turn-time 2.0      # seconds per LEFT/RIGHT
bash start_bridge.sh --max-drive 20       # auto-stop after 20 s without a new command
bash start_bridge.sh --link-timeout 2.0   # stop after 2 s without the laptop
python3 eeg_cmd_bridge.py --dry-run       # no ROS: just print what it would send
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Test says NO REPLY | robot IP, same Wi-Fi, bridge running (`listening ... port 9000`), Windows firewall |
| Bridge logs commands but the car doesn't move | chassis driver not running, or `ros2 topic info /cmd_vel` shows 0 subscribers; inside Docker? run the bridge in the same container |
| Car moves but steering is reversed | `--steer -0.6` |
| Car steers but doesn't drive during LEFT/RIGHT while stopped | expected only if `--speed` is 0 |
| Robot keeps stopping by itself | Wi-Fi drops heartbeats: move closer / raise `--link-timeout` |
