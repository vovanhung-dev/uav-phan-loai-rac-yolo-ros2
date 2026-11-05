import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def lawnmower_waypoints(area_half, lane_spacing, altitude):
    waypoints = []
    lanes = int(round(2 * area_half / lane_spacing)) + 1
    for lane in range(lanes):
        y = -area_half + lane * lane_spacing
        xs = (-area_half, area_half) if lane % 2 == 0 else (area_half, -area_half)
        waypoints.append((xs[0], y, altitude))
        waypoints.append((xs[1], y, altitude))
    return waypoints


class UavPatrolNode(Node):

    def __init__(self):
        super().__init__('uav_patrol')
        self.declare_parameter('mode', 'scan')
        self.declare_parameter('altitude', 3.0)
        self.declare_parameter('area_half', 5.5)
        self.declare_parameter('lane_spacing', 1.8)
        self.declare_parameter('speed', 0.6)
        self.declare_parameter('hover_x', 0.0)
        self.declare_parameter('hover_y', 0.0)
        self.declare_parameter('dwell_seconds', 1.5)
        self.declare_parameter('loop', True)

        self.mode = self.get_parameter('mode').value
        self.altitude = float(self.get_parameter('altitude').value)
        self.speed = float(self.get_parameter('speed').value)
        self.dwell_seconds = float(self.get_parameter('dwell_seconds').value)
        self.loop = bool(self.get_parameter('loop').value)
        if self.mode == 'hover':
            self.waypoints = [(float(self.get_parameter('hover_x').value),
                               float(self.get_parameter('hover_y').value), self.altitude)]
        else:
            self.waypoints = lawnmower_waypoints(float(self.get_parameter('area_half').value),
                                                 float(self.get_parameter('lane_spacing').value), self.altitude)

        self.cmd_pub = self.create_publisher(Twist, '/uav/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, '/uav/enable', 10)
        self.create_subscription(Odometry, '/uav/odometry', self.on_odometry, 10)
        self.pose = None
        self.target_index = 0
        self.phase = 'takeoff'
        self.dwell_started = None
        self.create_timer(0.05, self.control_step)
        self.get_logger().info(f'Chế độ {self.mode}, {len(self.waypoints)} waypoint, độ cao {self.altitude} m')

    def on_odometry(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, p.z, yaw_from_quaternion(msg.pose.pose.orientation))

    def control_step(self):
        self.enable_pub.publish(Bool(data=True))
        if self.pose is None:
            return
        x, y, z, yaw = self.pose
        if self.phase == 'takeoff':
            if abs(z - self.altitude) < 0.15:
                self.phase = 'travel'
                self.get_logger().info('Đã cất cánh, bắt đầu di chuyển')
            target = (x, y, self.altitude)
        else:
            target = self.waypoints[self.target_index]

        ex, ey, ez = target[0] - x, target[1] - y, target[2] - z
        horizontal = math.hypot(ex, ey)
        gain = 0.8
        vx_world, vy_world = gain * ex, gain * ey
        norm = math.hypot(vx_world, vy_world)
        if norm > self.speed:
            vx_world, vy_world = vx_world * self.speed / norm, vy_world * self.speed / norm
        cmd = Twist()
        cmd.linear.x = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        cmd.linear.y = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world
        cmd.linear.z = max(-0.6, min(0.6, 1.0 * ez))
        cmd.angular.z = max(-0.5, min(0.5, -1.0 * math.atan2(math.sin(yaw), math.cos(yaw))))
        self.cmd_pub.publish(cmd)

        if self.phase != 'travel' or horizontal > 0.2 or abs(ez) > 0.2:
            self.dwell_started = None
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.dwell_started is None:
            self.dwell_started = now
            return
        if now - self.dwell_started < self.dwell_seconds:
            return
        self.dwell_started = None
        if self.target_index + 1 < len(self.waypoints):
            self.target_index += 1
        elif self.loop and len(self.waypoints) > 1:
            self.target_index = 0
            self.get_logger().info('Hoàn thành một vòng quét, quay lại điểm đầu')


def main(args=None):
    rclpy.init(args=args)
    node = UavPatrolNode()
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
