"""Minimal example client and end-to-end check of the ROS2 interface (run while the launch is up):"""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, Float64MultiArray, String

NS = "/mobile_panda"


def yaw_of(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


class Check(Node):
    def __init__(self):
        super().__init__("mobile_panda_teleop_check")
        self.pose = self.joints = self.scene = None
        self.images, self.infos = {}, {}
        self.create_subscription(PoseStamped, f"{NS}/base_pose", lambda m: setattr(self, "pose", m), qos_profile_sensor_data)
        self.create_subscription(JointState, f"{NS}/joint_states", lambda m: setattr(self, "joints", m), qos_profile_sensor_data)
        for cam in ("base_camera", "hand_camera"):
            self.create_subscription(Image, f"{NS}/{cam}/rgb", lambda m, c=cam: self.images.__setitem__(c, m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, f"{NS}/{cam}/camera_info", lambda m, c=cam: self.infos.__setitem__(c, m), qos_profile_sensor_data)
        sticky = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, f"{NS}/scene_info", lambda m: setattr(self, "scene", json.loads(m.data)), sticky)
        self.cmd = self.create_publisher(Twist, f"{NS}/cmd_vel", 1)
        self.arm = self.create_publisher(Float64MultiArray, f"{NS}/arm_joint_delta", 1)
        self.grip = self.create_publisher(Float64, f"{NS}/gripper", 1)

    def spin_for(self, seconds, every=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if every:
                every()
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait(self, predicate, seconds, what):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not predicate():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not predicate():
            raise RuntimeError(f"timed out waiting for {what}")

    def joint(self, name):
        return self.joints.position[list(self.joints.name).index(name)]

    def drive(self, linear, angular, seconds):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(linear), float(angular)
        self.spin_for(seconds, every=lambda: self.cmd.publish(msg))
        self.spin_for(1.5)

    def run(self):
        results = {}
        self.wait(lambda: self.pose and self.joints and len(self.images) == 2 and len(self.infos) == 2 and self.scene,
                  120, "base_pose, joint_states, both camera images/infos and scene_info")
        results["images"] = {k: f"{m.width}x{m.height} {m.encoding}" for k, m in self.images.items()}
        p0 = self.pose.pose.position
        x0, y0 = p0.x, p0.y
        self.drive(0.15, 0.0, 2.0)
        p = self.pose.pose.position
        results["forward_m"] = round(math.hypot(p.x - x0, p.y - y0), 3)
        yaw1 = yaw_of(self.pose.pose.orientation)
        self.drive(0.0, 0.4, 2.0)
        results["turn_rad"] = round(math.atan2(math.sin(yaw_of(self.pose.pose.orientation) - yaw1),
                                               math.cos(yaw_of(self.pose.pose.orientation) - yaw1)), 3)
        q1 = self.joint("panda_joint1")
        delta = Float64MultiArray(data=[0.02, 0, 0, 0, 0, 0, 0])
        self.spin_for(1.0, every=lambda: self.arm.publish(delta))
        self.spin_for(1.5)
        results["joint1_change_rad"] = round(self.joint("panda_joint1") - q1, 3)
        self.grip.publish(Float64(data=0.0))
        self.spin_for(2.0)
        results["finger_closed_m"] = round(self.joint("panda_finger_joint1"), 4)
        self.grip.publish(Float64(data=0.04))
        self.spin_for(2.0)
        results["finger_open_m"] = round(self.joint("panda_finger_joint1"), 4)
        ok = (results["forward_m"] > 0.2 and results["turn_rad"] > 0.5 and results["joint1_change_rad"] > 0.05
              and results["finger_closed_m"] < 0.005 and results["finger_open_m"] > 0.035)
        results["passed"] = ok
        print(json.dumps(results, indent=2), flush=True)
        return ok


def main():
    rclpy.init()
    node = Check()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
