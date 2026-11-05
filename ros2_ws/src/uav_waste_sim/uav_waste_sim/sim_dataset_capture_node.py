import json
import math
import os
from bisect import bisect_left
from collections import deque

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

from uav_waste_sim.waste_classes import WASTE_CLASSES

CAMERA_OFFSET = np.array([0.0, 0.0, -0.07])
CAMERA_PITCH = math.pi / 2
OBJECT_HEIGHT = 0.004


def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def load_object_outlines(layout, models_dir):
    outlines = []
    for obj in layout['objects']:
        texture_path = os.path.join(models_dir, obj['texture'].replace('model://', ''))
        rgba = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        alpha = rgba[:, :, 3] > 128
        contours, _ = cv2.findContours(alpha.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(np.float64)
        height, width = alpha.shape
        local_x = (hull[:, 0] / width - 0.5) * obj['size_x']
        local_y = (0.5 - hull[:, 1] / height) * obj['size_y']
        c, s = math.cos(obj['yaw']), math.sin(obj['yaw'])
        world = np.column_stack([
            obj['x'] + c * local_x - s * local_y,
            obj['y'] + s * local_x + c * local_y,
            np.full(len(local_x), OBJECT_HEIGHT),
        ])
        outlines.append((WASTE_CLASSES.index(obj['class_name']), world))
    return outlines


class SimDatasetCaptureNode(Node):

    def __init__(self):
        super().__init__('sim_dataset_capture')
        share_dir = get_package_share_directory('uav_waste_sim')
        self.declare_parameter('layout_path', os.path.join(share_dir, 'config', 'waste_layout.json'))
        self.declare_parameter('models_dir', os.path.join(share_dir, 'models'))
        self.declare_parameter('output_dir', os.path.expanduser('~/uav_data/datasets/gazebo_sim'))
        self.declare_parameter('split', 'test')
        self.declare_parameter('capture_period', 1.0)
        self.declare_parameter('max_images', 400)
        self.declare_parameter('min_visible_ratio', 0.3)
        self.declare_parameter('max_tilt_deg', 3.0)
        self.declare_parameter('min_altitude', 2.0)
        self.declare_parameter('horizontal_fov', 1.047)
        self.declare_parameter('save_debug', True)

        with open(self.get_parameter('layout_path').value, encoding='utf-8') as handle:
            layout = json.load(handle)
        self.outlines = load_object_outlines(layout, self.get_parameter('models_dir').value)
        output_dir = self.get_parameter('output_dir').value
        split = self.get_parameter('split').value
        self.image_dir = os.path.join(output_dir, 'images', split)
        self.label_dir = os.path.join(output_dir, 'labels', split)
        self.debug_dir = os.path.join(output_dir, 'debug', split)
        for path in (self.image_dir, self.label_dir, self.debug_dir):
            os.makedirs(path, exist_ok=True)
        self.capture_period = float(self.get_parameter('capture_period').value)
        self.max_images = int(self.get_parameter('max_images').value)
        self.min_visible_ratio = float(self.get_parameter('min_visible_ratio').value)
        self.max_tilt = math.radians(float(self.get_parameter('max_tilt_deg').value))
        self.min_altitude = float(self.get_parameter('min_altitude').value)
        self.horizontal_fov = float(self.get_parameter('horizontal_fov').value)
        self.save_debug = bool(self.get_parameter('save_debug').value)

        self.bridge = CvBridge()
        self.odometry = deque(maxlen=400)
        self.last_capture = -1e9
        self.saved = len(os.listdir(self.image_dir))
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
        self.create_subscription(Odometry, '/uav/odometry', self.on_odometry, 50)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, sensor_qos)
        self.get_logger().info(f'Ghi dữ liệu mô phỏng vào {output_dir} ({len(self.outlines)} vật thể)')

    def on_odometry(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odometry.append((stamp, np.array([p.x, p.y, p.z]), quaternion_to_matrix(q.x, q.y, q.z, q.w)))

    def pose_at(self, stamp):
        if not self.odometry:
            return None
        stamps = [entry[0] for entry in self.odometry]
        idx = min(bisect_left(stamps, stamp), len(stamps) - 1)
        candidates = [self.odometry[i] for i in (idx - 1, idx) if 0 <= i < len(stamps)]
        best = min(candidates, key=lambda entry: abs(entry[0] - stamp))
        return best if abs(best[0] - stamp) < 0.05 else None

    def project_labels(self, position, rotation, width, height):
        camera_rotation = rotation @ rotation_y(CAMERA_PITCH)
        camera_position = position + rotation @ CAMERA_OFFSET
        focal = (width / 2.0) / math.tan(self.horizontal_fov / 2.0)
        labels = []
        for class_id, world_points in self.outlines:
            local = (world_points - camera_position) @ camera_rotation
            if np.any(local[:, 0] <= 0.05):
                continue
            u = width / 2.0 - focal * local[:, 1] / local[:, 0]
            v = height / 2.0 - focal * local[:, 2] / local[:, 0]
            x1, x2, y1, y2 = u.min(), u.max(), v.min(), v.max()
            full_area = (x2 - x1) * (y2 - y1)
            cx1, cx2 = max(0.0, x1), min(float(width), x2)
            cy1, cy2 = max(0.0, y1), min(float(height), y2)
            if cx2 <= cx1 or cy2 <= cy1 or full_area <= 0:
                continue
            if (cx2 - cx1) * (cy2 - cy1) / full_area < self.min_visible_ratio:
                continue
            labels.append((class_id, cx1, cy1, cx2, cy2))
        return labels

    def on_image(self, msg):
        if self.saved >= self.max_images:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp - self.last_capture < self.capture_period:
            return
        pose = self.pose_at(stamp)
        if pose is None:
            return
        _, position, rotation = pose
        tilt = math.acos(max(-1.0, min(1.0, rotation[2, 2])))
        if tilt > self.max_tilt or position[2] < self.min_altitude:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        labels = self.project_labels(position, rotation, width, height)
        stem = f'sim_{self.saved:05d}'
        cv2.imwrite(os.path.join(self.image_dir, f'{stem}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(self.label_dir, f'{stem}.txt'), 'w', encoding='utf-8') as handle:
            for class_id, x1, y1, x2, y2 in labels:
                handle.write(f'{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} '
                             f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n')
        if self.save_debug and self.saved % 10 == 0:
            debug = frame.copy()
            for class_id, x1, y1, x2, y2 in labels:
                cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                cv2.putText(debug, WASTE_CLASSES[class_id], (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1)
            cv2.imwrite(os.path.join(self.debug_dir, f'{stem}.jpg'), debug)
        self.saved += 1
        self.last_capture = stamp
        if self.saved % 20 == 0:
            self.get_logger().info(f'Đã lưu {self.saved}/{self.max_images} ảnh')
        if self.saved >= self.max_images:
            self.get_logger().info('Hoàn tất thu thập dữ liệu mô phỏng')


def main(args=None):
    rclpy.init(args=args)
    node = SimDatasetCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
