#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$HOME/uav_venv/bin/activate"
cd "$PROJECT_DIR/ai"

python download_data.py
python make_cutouts.py
python prepare_real.py
python synth_aerial.py
python synth_aerial.py --plain --name synthetic_plain
python gen_gazebo_world.py

cd "$HOME/uav_ws" && colcon build --symlink-install && cd "$PROJECT_DIR/ai"
bash "$PROJECT_DIR/scripts/capture_sim_testset.sh"

python train.py
python evaluate.py
cp "$HOME/uav_data/runs/E5_yolo11s_mix/weights/best.pt" "$PROJECT_DIR/ros2_ws/src/uav_waste_sim/weights/waste_yolo.pt"
