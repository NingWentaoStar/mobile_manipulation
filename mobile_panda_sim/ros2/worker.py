"""Simulation worker: owns the MuJoCo scene. stdin/stdout carry newline-delimited JSON only."""
import argparse
import base64
import contextlib
import json
import math
import queue
import sys
import threading
import time
import traceback

from ..geometry import GRIPPER_RANGE
from .protocol import validate_command

VELOCITY_TIMEOUT = 0.3
PUBLISH_PERIOD = 0.1


def emit(stream, kind, **payload):
    stream.write(json.dumps({"type": kind, **payload}, allow_nan=False) + "\n")
    stream.flush()


def pose_dict(position, quaternion):
    return {"position": [float(v) for v in position], "quaternion_wxyz": [float(v) for v in quaternion]}


def yaw_of(q):
    return math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))


def jsonable(value):
    """Recursively convert numpy arrays and scalars so skill details can be sent as JSON."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


class Simulation:
    def __init__(self, args, send):
        from mobile_panda_sim.mj.sim import MujocoScene

        self.args, self.send = args, send
        self.trajectory = 0
        self.scene = MujocoScene(args.base, args.start, viewer=args.viewer)
        self.joint_names = self.scene.joint_names
        self.control_timestep = self.scene.control_timestep
        self.skills = None   # created on the first skill request (imports mplib/torch)
        self.reset()

    def reset(self):
        if self.skills is not None:
            self.skills.reset()
        else:
            self.scene.reset()
        self.trajectory += 1
        self.steps, self.sim_time = 0, 0.0
        self.velocity, self.velocity_deadline = (0.0, 0.0), 0.0
        self.arm_delta = None
        self.gripper = GRIPPER_RANGE[1]
        self.images, self.images_step = None, -1
        self.last_publish = 0.0
        self.last_skill_step = None

    def handle(self, cmd):
        validate_command(cmd)
        kind, now = cmd["kind"], time.monotonic()
        if kind == "stop":
            self.velocity, self.velocity_deadline, self.arm_delta = (0.0, 0.0), 0.0, None
            self.send("stopped", trajectory=self.trajectory)
            return
        if kind == "reset":
            self.reset()
            self.send("reset_complete", trajectory=self.trajectory)
            return
        if kind == "skill":
            self.run_skill(cmd)
            return
        if kind == "velocity":
            self.velocity = (float(cmd.get("linear", 0)), float(cmd.get("angular", 0)))
            self.velocity_deadline = now + VELOCITY_TIMEOUT
        elif kind == "arm":
            self.arm_delta = [float(x) for x in cmd["delta"]]
        elif kind == "gripper":
            self.gripper = float(cmd["width"])

    def tick(self):
        """Advance physics one control period; controllers hold when no command is pending."""
        now = time.monotonic()
        linear, angular = self.velocity if now <= self.velocity_deadline else (0.0, 0.0)
        self.scene.step(linear, angular, self.arm_delta, self.gripper)
        self.arm_delta = None
        self.steps += 1
        self.sim_time += self.control_timestep

    def run_skill(self, cmd):
        """Run a skill on this scene (skills drive physics themselves while they run)."""
        request_id = cmd.get("request_id")
        self.velocity, self.velocity_deadline, self.arm_delta = (0.0, 0.0), 0.0, None
        try:
            if self.skills is None:
                from mobile_panda_sim.skills import SkillLibrary
                self.skills = SkillLibrary(self.scene, on_step=self.skill_step, on_idle=self.publish_if_due)
            self.last_skill_step = time.monotonic()
            result = self.skills.run(cmd["name"], **(cmd.get("params") or {}))
            payload = dict(success=result.success, message=result.message, info=jsonable(result.info))
        except KeyError as error:
            payload = dict(success=False, message=str(error), info={})
        except Exception as error:
            payload = dict(success=False, message=f"{type(error).__name__}: {error}", info={})
        if self.skills is not None:
            # Keep the gripper where the skill left it; otherwise the next tick reopens it and drops the object.
            low, high = GRIPPER_RANGE
            self.gripper = low + (self.skills.context.gripper + 1) / 2 * (high - low)
        try:
            self.send("skill_result", request_id=request_id, **payload)
        except (TypeError, ValueError):
            self.send("skill_result", request_id=request_id, success=payload["success"],
                      message=payload["message"], info={})
        self.publish_if_due()

    def skill_step(self):
        """One control step taken by a skill (the skill drives the scene itself), paced in real time."""
        self.steps += 1
        self.sim_time += self.control_timestep
        self.publish_if_due()
        now = time.monotonic()
        if self.last_skill_step is not None:
            time.sleep(max(0.0, self.control_timestep - (now - self.last_skill_step)))
        self.last_skill_step = time.monotonic()

    def publish_if_due(self):
        now = time.monotonic()
        if now - self.last_publish >= PUBLISH_PERIOD:
            self.last_publish = now
            self.snapshot()

    def snapshot(self):
        scene = self.scene
        if self.images_step != self.steps:
            self.images = {name: {"height": int(a.shape[0]), "width": int(a.shape[1]),
                                  "data": base64.b64encode(a.tobytes()).decode("ascii")}
                           for name, a in scene.capture_rgb().items()}
            self.images_step = self.steps
        chassis = scene.pose(scene.chassis)
        qpos = scene.joint_positions().tolist()
        self.send("state", trajectory=self.trajectory, steps=self.steps, sim_time=round(self.sim_time, 4),
                  running=True, base=self.args.base, joint_names=self.joint_names,
                  images=self.images, base_pose={**pose_dict(*chassis), "yaw": yaw_of(chassis[1])},
                  root_pose=pose_dict(*scene.pose(scene.root)), tcp_pose=pose_dict(*scene.tcp_pose()),
                  joint_positions=qpos, joint_velocities=scene.joint_velocities().tolist(), gripper_width=qpos[-2:],
                  objects={name: pose_dict(*scene.pose(body)) for name, body in scene.objects.items()},
                  articulations=scene.articulation_positions())

    def close(self):
        self.scene.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["wheels", "planar"], default="wheels")
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    protocol_out = sys.stdout
    send = lambda kind, **payload: emit(protocol_out, kind, **payload)
    commands = queue.Queue(maxsize=256)

    def read_commands():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line), timeout=1)
            except (ValueError, queue.Full) as error:
                send("error", reason=str(error))
        commands.put({"kind": "shutdown"})

    sim = None
    with contextlib.redirect_stdout(sys.stderr):
        try:
            sim = Simulation(args, send)
            threading.Thread(target=read_commands, daemon=True).start()
            send("scene", layout=sim.scene.layout())
            send("ready", base=args.base, joint_names=sim.joint_names,
                 control_timestep=sim.control_timestep, cameras=sim.scene.camera_intrinsics())
            while True:
                start = time.monotonic()
                shutdown = False
                for _ in range(256):
                    try:
                        cmd = commands.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(cmd, dict) and cmd.get("kind") == "shutdown":
                        shutdown = True
                        break
                    try:
                        sim.handle(cmd)
                    except (ValueError, TypeError) as error:
                        send("rejected", reason=str(error), command=cmd.get("kind") if isinstance(cmd, dict) else None)
                if shutdown:
                    break
                sim.tick()
                sim.publish_if_due()
                time.sleep(max(0.0, sim.control_timestep - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
        except Exception as error:
            send("fatal", reason=f"{type(error).__name__}: {error}")
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            if sim is not None:
                sim.close()


if __name__ == "__main__":
    main()
