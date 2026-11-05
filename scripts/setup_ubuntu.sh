#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive

if [ ! -d /opt/ros/jazzy ]; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common curl
  sudo add-apt-repository -y universe
  VER=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-cv-bridge ros-jazzy-vision-msgs \
    ros-jazzy-rqt-image-view python3-colcon-common-extensions python3-pip python3-venv
fi

if [ -e /dev/dxg ]; then
  sudo usermod -aG video,render "$USER"
fi

if [ ! -d "$HOME/uav_venv" ]; then
  python3 -m venv --system-site-packages "$HOME/uav_venv"
fi
source "$HOME/uav_venv/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

source /opt/ros/jazzy/setup.bash
mkdir -p "$HOME/uav_ws"
ln -sfn "$PROJECT_DIR/ros2_ws/src" "$HOME/uav_ws/src"
cd "$HOME/uav_ws" && colcon build --symlink-install
echo "Cài đặt hoàn tất. Chạy demo: bash $PROJECT_DIR/scripts/run_demo.sh"
