#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source "$HOME/uav_venv/bin/activate"
source "$HOME/uav_ws/install/setup.bash"
ros2 launch uav_waste_sim sim.launch.py "$@"
