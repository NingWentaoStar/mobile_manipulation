"""ROS-facing process; runs on system Python and never imports MuJoCo."""
import base64
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

NS = "/mobile_panda"
CAMERAS = ("base_camera", "hand_camera")
CONDA_ROOTS = ("anaconda3", "miniconda3", "miniforge3")


def find_python(root):
    """The repository's .venv, or else a Conda environment with mujoco, preferring one that also has mplib."""
    venv = root / ".venv/bin/python"
    if venv.is_file():
        return venv
    found = []
    for conda in CONDA_ROOTS:
        for env in sorted((Path.home() / conda / "envs").glob("*")):
            found += [(env, site) for site in env.glob("lib/python3.1[0-9]/site-packages") if any(site.glob("mujoco-*"))]
    if not found:
        raise RuntimeError("no Python with mujoco found; create .venv (see README) or set sim_python")
    return max(found, key=lambda item: any(item[1].glob("mplib-*")))[0] / "bin/python"


class Bridge(Node):
    def __init__(self):
        super().__init__("mobile_panda_bridge")
        defaults = {"project_root": os.environ.get("MOBILE_PANDA_ROOT", str(Path.cwd())),
                    "sim_python": os.environ.get("MOBILE_PANDA_SIM_PYTHON", ""),
                    "base": "wheels", "viewer": True,
                    "start_x": float("nan"), "start_y": float("nan"), "start_yaw": float("nan")}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        param = lambda name: self.get_parameter(name).value
        root = Path(param("project_root")).expanduser().resolve()
        python = Path(param("sim_python")).expanduser() if param("sim_python") else find_python(root)
        self.get_logger().info(f"Simulation interpreter: {python}")
        if not (root / "skills/ros2/worker.py").is_file():
            raise RuntimeError(f"project_root={root} is not the repository root (skills/ missing)")
        if not python.is_file():
            raise RuntimeError(f"sim_python={python} not found; set it to an interpreter with mujoco")

        sticky = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, f"{NS}/state", sticky)
        self.status_pub = self.create_publisher(String, f"{NS}/status", sticky)
        self.scene_pub = self.create_publisher(String, f"{NS}/scene_info", sticky)
        self.skill_result_pub = self.create_publisher(String, f"{NS}/skill_result", 10)
        self.pose_pub = self.create_publisher(PoseStamped, f"{NS}/base_pose", qos_profile_sensor_data)
        self.joints_pub = self.create_publisher(JointState, f"{NS}/joint_states", qos_profile_sensor_data)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.image_pubs = {name: self.create_publisher(Image, f"{NS}/{name}/rgb", qos_profile_sensor_data)
                           for name in CAMERAS}
        self.info_pubs = {name: self.create_publisher(CameraInfo, f"{NS}/{name}/camera_info", qos_profile_sensor_data)
                          for name in CAMERAS}
        self.create_subscription(Twist, f"{NS}/cmd_vel", self.on_velocity, 1)
        self.create_subscription(Float64MultiArray, f"{NS}/arm_joint_delta", self.on_arm, 1)
        self.create_subscription(Float64, f"{NS}/gripper", self.on_gripper, 1)
        self.create_subscription(String, f"{NS}/skill_request", self.on_skill_request, 10)
        self.create_service(Trigger, f"{NS}/stop", lambda req, res: self.trigger("stop", res))
        self.create_service(Trigger, f"{NS}/reset", lambda req, res: self.trigger("reset", res))

        self.outbox, self.events = queue.Queue(maxsize=256), queue.Queue(maxsize=128)
        self.latest_state, self.lock = None, threading.Lock()
        self.ready, self.closing, self.intrinsics = False, False, {}
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = str(root)
        args = [str(python), "-u", "-m", "skills.ros2.worker", "--base", param("base")]
        start = [param("start_x"), param("start_y"), param("start_yaw")]
        if not any(v != v for v in start):
            args += ["--start", *map(str, start)]
        if param("viewer"):
            args.append("--viewer")
        self.process = subprocess.Popen(args, cwd=root, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self.read_worker, daemon=True).start()
        threading.Thread(target=self.write_worker, daemon=True).start()
        self.create_timer(0.02, self.poll)
        self.parent = os.getppid()
        self.create_timer(1.0, self.check_parent)
        self.status_pub.publish(String(data=json.dumps({"type": "starting"})))
        self.get_logger().info("Loading scene; physics runs continuously, commands override the current targets")

    def send(self, command):
        if not self.ready or self.process.poll() is not None:
            raise ValueError("simulation is not ready")
        try:
            self.outbox.put_nowait(command)
        except queue.Full:
            raise ValueError("command queue full; reduce publish rate")

    def reject(self, reason):
        self.get_logger().warn(reason, throttle_duration_sec=1.0)
        self.status_pub.publish(String(data=json.dumps({"type": "rejected", "reason": reason})))

    def on_velocity(self, msg):
        if any(abs(x) > 1e-9 for x in (msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y)):
            return self.reject("differential base accepts linear.x and angular.z only")
        try:
            self.send({"kind": "velocity", "linear": msg.linear.x, "angular": msg.angular.z})
        except ValueError as error:
            self.reject(str(error))

    def on_arm(self, msg):
        try:
            self.send({"kind": "arm", "delta": list(msg.data)})
        except ValueError as error:
            self.reject(str(error))

    def on_gripper(self, msg):
        try:
            self.send({"kind": "gripper", "width": msg.data})
        except ValueError as error:
            self.reject(str(error))

    def on_skill_request(self, msg):
        try:
            request = json.loads(msg.data)
        except ValueError as error:
            request, problem = None, f"skill request is not JSON: {error}"
        else:
            problem = None if isinstance(request, dict) and isinstance(request.get("name"), str) \
                else "skill request needs a JSON object with a string `name`"
        if problem:
            self.publish_skill_result({"request_id": request.get("request_id") if isinstance(request, dict) else None,
                                       "success": False, "message": problem, "info": {}})
            return
        try:
            self.send({"kind": "skill", **request})
        except ValueError as error:
            self.publish_skill_result({"request_id": request.get("request_id"), "success": False,
                                       "message": str(error), "info": {}})

    def publish_skill_result(self, payload):
        self.skill_result_pub.publish(String(data=json.dumps(payload)))

    def trigger(self, kind, response):
        try:
            self.send({"kind": kind})
            response.success, response.message = True, f"queued; completion is reported on {NS}/status"
        except ValueError as error:
            response.success, response.message = False, str(error)
        return response

    def write_worker(self):
        try:
            while not self.closing:
                command = self.outbox.get()
                if command is None:
                    return
                self.process.stdin.write(json.dumps(command) + "\n")
                self.process.stdin.flush()
        except (BrokenPipeError, ValueError):
            return

    def read_worker(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("type") == "state":
                    with self.lock:
                        self.latest_state = message
                else:
                    self.events.put(message)
        finally:
            self.events.put({"type": "worker_exited", "returncode": self.process.poll()})

    def check_parent(self):
        if os.getppid() != self.parent:
            self.get_logger().warning("Parent process exited; shutting down")
            raise KeyboardInterrupt

    def poll(self):
        while True:
            try:
                message = self.events.get_nowait()
            except queue.Empty:
                break
            kind = message.get("type")
            if kind == "scene":
                self.scene_pub.publish(String(data=json.dumps(message["layout"])))
                continue
            if kind == "ready":
                self.ready, self.intrinsics = True, message.get("cameras", {})
                self.get_logger().info("Scene ready; command topics: "
                                       f"{NS}/cmd_vel, {NS}/arm_joint_delta, {NS}/gripper; "
                                       f"skills: {NS}/skill_request")
            elif kind in ("fatal", "worker_exited"):
                self.ready = False
                self.get_logger().error(json.dumps(message))
            elif kind == "rejected":
                self.get_logger().warn(message.get("reason", ""), throttle_duration_sec=1.0)
            elif kind == "skill_result":
                self.publish_skill_result(message)
                self.get_logger().info(f"skill [{message.get('request_id')}]: success={message.get('success')} "
                                       f"{message.get('message', '')}")
                continue
            self.status_pub.publish(String(data=json.dumps(message)))
        with self.lock:
            state, self.latest_state = self.latest_state, None
        if state is not None:
            self.publish_state(state)

    def publish_state(self, state):
        images = state.pop("images", None) or {}
        stamp = self.get_clock().now().to_msg()
        self.state_pub.publish(String(data=json.dumps(state)))

        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = stamp, "map"
        base = state["base_pose"]
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = base["position"]
        o = pose.pose.orientation
        o.w, o.x, o.y, o.z = base["quaternion_wxyz"]
        self.pose_pub.publish(pose)

        root = state["root_pose"]
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id = stamp, "map"
        tf.child_frame_id = "world_root" if state.get("base") == "planar" else "chassis_link"
        t, r = tf.transform.translation, tf.transform.rotation
        t.x, t.y, t.z = root["position"]
        r.w, r.x, r.y, r.z = root["quaternion_wxyz"]
        self.tf_broadcaster.sendTransform(tf)

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = state["joint_names"]
        joints.position, joints.velocity = state["joint_positions"], state["joint_velocities"]
        self.joints_pub.publish(joints)

        for name, image in images.items():
            if name not in self.image_pubs:
                continue
            frame = f"{name}_optical_frame"
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, frame
            msg.height, msg.width = image["height"], image["width"]
            msg.encoding, msg.is_bigendian, msg.step = "rgb8", 0, image["width"] * 3
            msg.data = base64.b64decode(image["data"])
            self.image_pubs[name].publish(msg)
            k = self.intrinsics.get(name)
            if k:
                info = CameraInfo()
                info.header = msg.header
                info.height, info.width = msg.height, msg.width
                info.distortion_model, info.d = "plumb_bob", [0.0] * 5
                info.k = [float(v) for row in k for v in row]
                info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.p = [k[0][0], k[0][1], k[0][2], 0.0, k[1][0], k[1][1], k[1][2], 0.0, 0.0, 0.0, 1.0, 0.0]
                self.info_pubs[name].publish(info)

    def destroy_node(self):
        self.closing = True
        try:
            self.outbox.put_nowait(None)
        except queue.Full:
            pass
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = None
    try:
        node = Bridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
