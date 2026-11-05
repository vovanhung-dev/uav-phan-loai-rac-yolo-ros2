import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DemoRecorderNode(Node):

    def __init__(self):
        super().__init__('demo_recorder')
        self.declare_parameter('topic', '/waste_detection/dashboard')
        self.declare_parameter('output_path', os.path.expanduser(f'~/uav_waste_demo_{int(time.time())}.mp4'))
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('duration', 60.0)
        self.output_path = self.get_parameter('output_path').value
        self.fps = float(self.get_parameter('fps').value)
        self.duration = float(self.get_parameter('duration').value)
        self.bridge = CvBridge()
        self.writer = None
        self.started = None
        self.frames = 0
        self.done = False
        self.create_subscription(Image, self.get_parameter('topic').value, self.on_image, 10)
        self.get_logger().info(f'Ghi video vào {self.output_path} trong {self.duration:.0f} s')

    def on_image(self, msg):
        if self.done:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            self.started = time.time()
        self.writer.write(frame)
        self.frames += 1
        if time.time() - self.started >= self.duration:
            self.finish()

    def finish(self):
        if self.writer is not None and not self.done:
            self.writer.release()
            self.get_logger().info(f'Đã lưu {self.frames} khung hình vào {self.output_path}')
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorderNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
