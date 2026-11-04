#!/usr/bin/env bash
# Start the EEG command bridge on the Yahboom RosmasterPi A1.
# Run it in the same ROS 2 environment where the chassis driver runs
# (inside Yahboom's docker container if your image uses one - it must use --net=host).
#
#   bash start_bridge.sh                  # default speed 0.15 m/s
#   bash start_bridge.sh --speed 0.2      # any eeg_cmd_bridge.py option can be added
set -e
cd "$(dirname "$0")"
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f ~/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash ] && source ~/yahboomcar_ros2_ws/yahboomcar_ws/install/setup.bash
echo "Robot IP address(es) - type one of these into MuseRobot.bat on the laptop:"
hostname -I
python3 eeg_cmd_bridge.py "$@"
