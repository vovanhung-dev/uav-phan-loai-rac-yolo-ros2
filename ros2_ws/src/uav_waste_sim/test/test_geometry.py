import math

import numpy as np

from uav_waste_sim.sim_dataset_capture_node import CAMERA_OFFSET, CAMERA_PITCH, quaternion_to_matrix, rotation_y
from uav_waste_sim.uav_patrol_node import lawnmower_waypoints, yaw_from_quaternion


class FakeQuaternion:

    def __init__(self, yaw):
        self.x = 0.0
        self.y = 0.0
        self.z = math.sin(yaw / 2)
        self.w = math.cos(yaw / 2)


def project(point, uav_position, uav_rotation, width=640, height=480, fov=1.047):
    camera_rotation = uav_rotation @ rotation_y(CAMERA_PITCH)
    camera_position = uav_position + uav_rotation @ CAMERA_OFFSET
    local = (np.asarray(point) - camera_position) @ camera_rotation
    focal = (width / 2) / math.tan(fov / 2)
    return width / 2 - focal * local[1] / local[0], height / 2 - focal * local[2] / local[0]


def test_lawnmower_covers_area_and_alternates_direction():
    waypoints = lawnmower_waypoints(5.0, 2.5, 3.0)
    assert len(waypoints) == 10
    assert waypoints[0] == (-5.0, -5.0, 3.0)
    assert waypoints[1] == (5.0, -5.0, 3.0)
    assert waypoints[2] == (5.0, -2.5, 3.0)
    assert all(z == 3.0 for _, _, z in waypoints)


def test_yaw_from_quaternion_roundtrip():
    for yaw in (-2.5, -0.3, 0.0, 1.2, 3.0):
        assert math.isclose(yaw_from_quaternion(FakeQuaternion(yaw)), yaw, abs_tol=1e-9)


def test_point_below_camera_projects_to_image_center():
    position = np.array([1.0, 2.0, 3.0])
    u, v = project([1.0, 2.0, 0.0], position, np.eye(3))
    assert math.isclose(u, 320, abs_tol=1e-6)
    assert math.isclose(v, 240, abs_tol=1e-6)


def test_forward_point_appears_above_center_and_left_point_left():
    position = np.array([0.0, 0.0, 3.0])
    _, v_forward = project([0.5, 0.0, 0.0], position, np.eye(3))
    u_left, _ = project([0.0, 0.5, 0.0], position, np.eye(3))
    assert v_forward < 240
    assert u_left < 320


def test_quaternion_matrix_matches_yaw_rotation():
    yaw = 0.7
    q = FakeQuaternion(yaw)
    matrix = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    expected = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    assert np.allclose(matrix, expected)
