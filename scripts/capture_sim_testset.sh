#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source "$HOME/uav_venv/bin/activate"
source "$HOME/uav_ws/install/setup.bash"
if [ -e /dev/dxg ]; then export GALLIUM_DRIVER=d3d12; fi
capture() {
  ALT=$1; OUT=$2; N=$3; T=$4
  rm -rf $OUT
  timeout $T ros2 launch uav_waste_sim sim.launch.py headless:=true detector:=false monitor:=false rviz:=false altitude:=$ALT > /tmp/sim_$ALT.log 2>&1 &
  sleep 12
  timeout $((T-15)) ros2 run uav_waste_sim sim_dataset_capture --ros-args -p output_dir:=$OUT -p max_images:=$N -p capture_period:=0.7 -p min_altitude:=$(python3 -c "print($ALT-0.5)") > /tmp/cap_$ALT.log 2>&1
  wait
  echo "ALT $ALT: $(ls $OUT/images/test | wc -l) ảnh, $(cat $OUT/labels/test/*.txt | wc -l) nhãn"
}
capture 3.0 ~/uav_data/datasets/gazebo_sim 300 420
capture 2.0 ~/uav_data/datasets/gazebo_alt2 120 300
capture 4.0 ~/uav_data/datasets/gazebo_alt4 120 300
