import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def venv_python_path():
    venv_dir = os.environ.get('UAV_VENV', os.path.expanduser('~/uav_venv'))
    site_packages = glob.glob(os.path.join(venv_dir, 'lib', 'python3*', 'site-packages'))
    if not site_packages:
        return []
    return [AppendEnvironmentVariable('PYTHONPATH', site_packages[0], prepend=True)]


def wsl_gpu_rendering():
    if os.path.exists('/dev/dxg') and 'GALLIUM_DRIVER' not in os.environ:
        return [SetEnvironmentVariable('GALLIUM_DRIVER', 'd3d12')]
    return []


def generate_launch_description():
    share_dir = get_package_share_directory('uav_waste_sim')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')
    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')

    arguments = [
        DeclareLaunchArgument('world', default_value=os.path.join(share_dir, 'worlds', 'waste_field.sdf')),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('monitor', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('patrol', default_value='true'),
        DeclareLaunchArgument('patrol_mode', default_value='scan'),
        DeclareLaunchArgument('altitude', default_value='3.0'),
        DeclareLaunchArgument('model_path', default_value=os.path.join(share_dir, 'weights', 'waste_yolo.pt')),
        DeclareLaunchArgument('confidence', default_value='0.35'),
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [world, ' -r -v 2'], 'on_exit_shutdown': 'true'}.items(),
        condition=UnlessCondition(headless),
    )
    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [world, ' -r -s -v 2'], 'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(headless),
    )

    follow_uav = [
        TimerAction(
            period=delay,
            condition=UnlessCondition(headless),
            actions=[
                ExecuteProcess(cmd=['gz', 'service', '-s', '/gui/follow/offset', '--reqtype', 'gz.msgs.Vector3d',
                                    '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
                                    '--req', 'x: -3.5, y: -3.5, z: 3.5'], output='log'),
                ExecuteProcess(cmd=['gz', 'service', '-s', '/gui/follow', '--reqtype', 'gz.msgs.StringMsg',
                                    '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
                                    '--req', 'data: "x3_uav"'], output='log'),
            ],
        )
        for delay in (15.0, 35.0)
    ]

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge', output='screen',
        parameters=[{'config_file': os.path.join(share_dir, 'config', 'bridge.yaml'), 'use_sim_time': True}],
    )
    detector = Node(
        package='uav_waste_sim', executable='waste_detector', name='waste_detector', output='screen',
        condition=IfCondition(LaunchConfiguration('detector')),
        parameters=[{'model_path': LaunchConfiguration('model_path'),
                     'confidence_threshold': LaunchConfiguration('confidence')}],
    )
    monitor = Node(
        package='uav_waste_sim', executable='waste_monitor', name='waste_monitor', output='screen',
        condition=IfCondition(LaunchConfiguration('monitor')),
    )
    patrol = Node(
        package='uav_waste_sim', executable='uav_patrol', name='uav_patrol', output='screen',
        condition=IfCondition(LaunchConfiguration('patrol')),
        parameters=[{'mode': LaunchConfiguration('patrol_mode'), 'altitude': LaunchConfiguration('altitude'),
                     'use_sim_time': True}],
    )
    odom_frame = Node(
        package='tf2_ros', executable='static_transform_publisher', name='world_to_odom',
        arguments=['--frame-id', 'world', '--child-frame-id', 'x3_uav/odom'],
    )
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(share_dir, 'config', 'waste_detection.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(arguments + venv_python_path() + wsl_gpu_rendering() + [
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(share_dir, 'models')),
        gazebo, gazebo_headless, *follow_uav, bridge, odom_frame, detector, monitor, patrol, rviz,
    ])
