import os
from glob import glob

from setuptools import setup

package_name = 'uav_waste_sim'


def collect_tree(source_dir):
    entries = []
    for root, _, files in os.walk(source_dir):
        if files:
            target = os.path.join('share', package_name, root)
            entries.append((target, [os.path.join(root, f) for f in files]))
    return entries


setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/weights', glob('weights/*.pt')),
    ] + collect_tree('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vovanhung-dev',
    maintainer_email='vohung77h@gmail.com',
    description='Mô phỏng UAV gắn camera và nhận diện rác bằng YOLO qua ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'waste_detector = uav_waste_sim.waste_detector_node:main',
            'waste_monitor = uav_waste_sim.waste_monitor_node:main',
            'uav_patrol = uav_waste_sim.uav_patrol_node:main',
            'sim_dataset_capture = uav_waste_sim.sim_dataset_capture_node:main',
            'demo_recorder = uav_waste_sim.demo_recorder_node:main',
        ],
    },
)
