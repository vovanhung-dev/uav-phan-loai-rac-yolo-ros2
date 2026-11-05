import json
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PilImage, ImageDraw, ImageFont
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from uav_waste_sim.waste_classes import WASTE_CLASSES, color_for

VIETNAMESE_NAMES = {
    'bottle': 'Chai',
    'plastic_bag': 'Túi nilon',
    'battery': 'Pin',
    'organic_waste': 'Rác hữu cơ',
}

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class WasteMonitorNode(Node):

    def __init__(self):
        super().__init__('waste_monitor')
        self.declare_parameter('snapshot_dir', os.path.expanduser('~/uav_waste_snapshots'))
        self.declare_parameter('show_window', True)
        self.snapshot_dir = self.get_parameter('snapshot_dir').value
        self.show_window = bool(self.get_parameter('show_window').value)

        self.bridge = CvBridge()
        self.raw_frame = None
        self.annotated_frame = None
        self.summary = {'counts': {}, 'fps': 0.0, 'latency_ms': 0.0, 'detections': []}
        self.peak_counts = {name: 0 for name in WASTE_CLASSES}
        self.title_font = load_font(22)
        self.body_font = load_font(17)
        self.small_font = load_font(14)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/camera/image_raw', self.on_raw, sensor_qos)
        self.create_subscription(Image, '/waste_detection/image_annotated', self.on_annotated, sensor_qos)
        self.create_subscription(String, '/waste_detection/summary', self.on_summary, 10)
        self.dashboard_pub = self.create_publisher(Image, '/waste_detection/dashboard', 10)
        self.create_timer(1.0 / 15.0, self.render)
        self.get_logger().info('Monitor đã sẵn sàng. Phím s: chụp ảnh, q: thoát cửa sổ.')

    def on_raw(self, msg):
        self.raw_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def on_annotated(self, msg):
        self.annotated_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def on_summary(self, msg):
        self.summary = json.loads(msg.data)
        for name, count in self.summary.get('counts', {}).items():
            if name in self.peak_counts:
                self.peak_counts[name] = max(self.peak_counts[name], count)

    def build_panel(self, height):
        width = 300
        panel = PilImage.new('RGB', (width, height), (24, 26, 30))
        draw = ImageDraw.Draw(panel)
        draw.text((16, 14), 'Giám sát rác thải UAV', font=self.title_font, fill=(255, 255, 255))
        draw.text((16, 46), f"FPS: {self.summary.get('fps', 0):.1f}   Trễ: {self.summary.get('latency_ms', 0):.0f} ms",
                  font=self.small_font, fill=(170, 175, 185))
        y = 84
        counts = self.summary.get('counts', {})
        for name in WASTE_CLASSES:
            b, g, r = color_for(name)
            draw.rectangle([16, y + 4, 30, y + 18], fill=(r, g, b))
            draw.text((40, y), f'{VIETNAMESE_NAMES[name]}', font=self.body_font, fill=(235, 235, 235))
            draw.text((190, y), f'{counts.get(name, 0)}', font=self.body_font, fill=(255, 255, 255))
            draw.text((230, y + 2), f'max {self.peak_counts[name]}', font=self.small_font, fill=(140, 145, 155))
            y += 34
        y += 8
        draw.line([16, y, width - 16, y], fill=(60, 64, 72), width=1)
        y += 12
        draw.text((16, y), 'Đối tượng trong khung:', font=self.small_font, fill=(170, 175, 185))
        y += 24
        detections = sorted(self.summary.get('detections', []), key=lambda d: -d['confidence'])
        for det in detections[:10]:
            line = f"{VIETNAMESE_NAMES.get(det['class_name'], det['class_name'])}  {det['confidence'] * 100:.0f}%"
            draw.text((24, y), line, font=self.small_font, fill=(220, 220, 220))
            y += 20
            if y > height - 24:
                break
        return cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)

    @staticmethod
    def caption(frame, text):
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        right = frame.shape[1]
        cv2.rectangle(frame, (right - width - 14, 0), (right, 26), (30, 30, 30), -1)
        cv2.putText(frame, text, (right - width - 7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return frame

    def render(self):
        if self.annotated_frame is None and self.raw_frame is None:
            return
        main_view = self.annotated_frame if self.annotated_frame is not None else self.raw_frame
        height = main_view.shape[0]
        views = []
        if self.raw_frame is not None:
            raw_small = cv2.resize(self.raw_frame, (int(self.raw_frame.shape[1] * height / self.raw_frame.shape[0]), height))
            views.append(self.caption(raw_small, 'Camera UAV (raw)'))
            views.append(np.full((height, 4, 3), 24, dtype=np.uint8))
        views.append(self.caption(main_view.copy(), 'YOLO'))
        views.append(self.build_panel(height))
        dashboard = cv2.hconcat(views)
        msg = self.bridge.cv2_to_imgmsg(dashboard, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.dashboard_pub.publish(msg)

        if not self.show_window:
            return
        cv2.imshow('UAV Waste Monitor', dashboard)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            os.makedirs(self.snapshot_dir, exist_ok=True)
            path = os.path.join(self.snapshot_dir, f'snapshot_{int(time.time())}.png')
            cv2.imwrite(path, dashboard)
            self.get_logger().info(f'Đã lưu {path}')
        elif key == ord('q'):
            self.show_window = False
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = WasteMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
