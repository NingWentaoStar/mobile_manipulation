"""Run one skill on the already running simulation.

    ros2 run mobile_panda_bridge run_skill navigate --param target='[-2.95, -1.15, 3.1416]'
    ros2 run mobile_panda_bridge run_skill pick --param object=lemon
    ros2 run mobile_panda_bridge run_skill open --param articulation=kitchen_counter --param joint=middle_slide_top

`--param VALUE` is parsed as JSON when possible, so numbers and lists work too.
"""
import argparse
import json
import sys
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String

NS = "/mobile_panda"


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="skill name: navigate, pick, place, open, close, reset_arm, suggest_base_pose, rl_pick_move, rl_place")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="skill parameter; VALUE is parsed as JSON when possible (repeatable)")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the result")
    parser.add_argument("--ready-timeout", type=float, default=30.0, help="seconds to wait for the simulation")
    return parser.parse_args(argv)


class RunSkill(Node):
    def __init__(self, request, args):
        super().__init__("run_skill")
        self.request, self.args = request, args
        self.result = None
        self.ready = False
        self.sent = None
        self.started = time.monotonic()
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(String, f"{NS}/skill_request", 10)
        self.create_subscription(String, f"{NS}/status", self.on_status, status_qos)
        self.create_subscription(String, f"{NS}/skill_result", self.on_result, 10)

    def on_status(self, msg):
        try:
            message = json.loads(msg.data)
        except ValueError:
            return
        if message.get("type") == "ready":
            self.ready = True

    def on_result(self, msg):
        try:
            message = json.loads(msg.data)
        except ValueError:
            return
        if message.get("request_id") == self.request["request_id"]:
            self.result = message

    def run(self):
        while rclpy.ok() and self.result is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.monotonic()
            if not self.ready:
                if now - self.started > self.args.ready_timeout:
                    print("simulation not ready; start it with "
                          "`ros2 launch mobile_panda_bridge simulation.launch.py`", file=sys.stderr)
                    return 2
                continue
            if self.sent is None:
                if self.publisher.get_subscription_count() > 0:
                    self.publisher.publish(String(data=json.dumps(self.request)))
                    self.sent = now
            elif now - self.sent > self.args.timeout:
                print(f"no skill result after {self.args.timeout:.0f} s", file=sys.stderr)
                return 2
        if self.result is None:
            return 2
        print(json.dumps({key: self.result.get(key) for key in ("success", "message", "info")},
                         ensure_ascii=False, indent=2))
        return 0 if self.result.get("success") else 1


def main():
    args = parse_args(remove_ros_args(sys.argv)[1:])
    params = {}
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"--param needs KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        params[key] = parse_value(value)
    request = {"request_id": uuid.uuid4().hex, "name": args.name, "params": params}
    rclpy.init()
    node = RunSkill(request, args)
    try:
        code = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
