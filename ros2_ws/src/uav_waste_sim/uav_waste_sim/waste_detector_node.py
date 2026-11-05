import json
import os
import time
from collections import deque

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from uav_waste_sim.waste_classes import color_for


def default_model_path():
    share_dir = get_package_share_directory('uav_waste_sim')
    return os.path.join(share_dir, 'weights', 'waste_yolo.pt')


def draw_detections(frame, detections, fps, latency_ms):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = color_for(det['class_name'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(frame, caption, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    hud = f'YOLO | {len(detections)} obj | {fps:.1f} FPS | {latency_ms:.0f} ms'
    bottom = frame.shape[0]
    cv2.rectangle(frame, (0, bottom - 26), (len(hud) * 9 + 12, bottom), (30, 30, 30), -1)
    cv2.putText(frame, hud, (6, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class WasteDetectorNode(Node):

    def __init__(self):
        super().__init__('waste_detector')
        self.declare_parameter('model_path', default_model_path())
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', '')

        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.image_size = int(self.get_parameter('image_size').value)
        device = self.get_parameter('device').value

        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device or ('cuda:0' if self._cuda_available() else 'cpu')
        self.class_names = self.model.names
        self.get_logger().info(f'Đã nạp model {model_path} trên {self.device}, lớp: {self.class_names}')

        self.bridge = CvBridge()
        self.frame_times = deque(maxlen=30)
        self.frame_count = 0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        input_topic = self.get_parameter('input_topic').value
        self.create_subscription(Image, input_topic, self.on_image, sensor_qos)

        self.annotated_pub = self.create_publisher(Image, '/waste_detection/image_annotated', 10)
        self.detections_pub = self.create_publisher(Detection2DArray, '/waste_detection/detections', 10)
        self.summary_pub = self.create_publisher(String, '/waste_detection/summary', 10)
        self.get_logger().info(f'Đang lắng nghe ảnh từ {input_topic}')

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        started = time.perf_counter()
        result = self.model.predict(frame, conf=self.confidence_threshold, iou=self.iou_threshold,
                                    imgsz=self.image_size, device=self.device, verbose=False)[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        now = time.perf_counter()
        self.frame_times.append(now)
        fps = 0.0
        if len(self.frame_times) > 1:
            fps = (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            detections.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'confidence': float(box.conf[0]),
                'bbox': [x1, y1, x2, y2],
            })

        self.publish_detections(msg.header, detections)
        annotated = draw_detections(frame.copy(), detections, fps, latency_ms)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header = msg.header
        self.annotated_pub.publish(annotated_msg)

        counts = {}
        for det in detections:
            counts[det['class_name']] = counts.get(det['class_name'], 0) + 1
        summary = {
            'frame': self.frame_count,
            'fps': round(fps, 2),
            'latency_ms': round(latency_ms, 1),
            'counts': counts,
            'detections': detections,
        }
        self.summary_pub.publish(String(data=json.dumps(summary)))

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Khung {self.frame_count}: {counts} | {fps:.1f} FPS | {latency_ms:.0f} ms')

    def publish_detections(self, header, detections):
        array_msg = Detection2DArray()
        array_msg.header = header
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            detection = Detection2D()
            detection.header = header
            detection.bbox.center.position.x = (x1 + x2) / 2.0
            detection.bbox.center.position.y = (y1 + y2) / 2.0
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['class_name']
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)
            array_msg.detections.append(detection)
        self.detections_pub.publish(array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WasteDetectorNode()
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
