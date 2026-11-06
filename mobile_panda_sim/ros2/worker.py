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
SETTLE_SECONDS = 1.0
PUBLISH_PERIOD = 0.1


def emit(stream, kind, **payload):
    stream.write(json.dumps({"type": kind, **payload}, allow_nan=False) + "\n")
    stream.flush()


def pose_dict(position, quaternion):
    return {"position": [float(v) for v in position], "quaternion_wxyz": [float(v) for v in quaternion]}


def yaw_of(q):
    return math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))


class Simulation:
    def __init__(self, args, send):
        from mobile_panda_sim.mj.sim import MujocoScene

        self.args, self.send = args, send
        self.trajectory = 0
        self.scene = MujocoScene(args.base, args.start, viewer=args.viewer)
        self.joint_names = self.scene.joint_names
        self.control_timestep = self.scene.control_timestep
        self.reset()

    def reset(self):
        self.scene.reset()
        self.trajectory += 1
        self.steps, self.sim_time = 0, 0.0
        self.velocity, self.velocity_deadline = (0.0, 0.0), 0.0
        self.arm_delta = None
        self.gripper = GRIPPER_RANGE[1]
        self.active_until = 0.0
        self.images, self.images_step = None, -1

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
        if kind == "velocity":
            self.velocity = (float(cmd.get("linear", 0)), float(cmd.get("angular", 0)))
            self.velocity_deadline = now + VELOCITY_TIMEOUT
        elif kind == "arm":
            self.arm_delta = [float(x) for x in cmd["delta"]]
        elif kind == "gripper":
            self.gripper = float(cmd["width"])
        self.active_until = now + SETTLE_SECONDS

    def tick(self):
        now = time.monotonic()
        if now > self.active_until:
            return False
        linear, angular = self.velocity if now <= self.velocity_deadline else (0.0, 0.0)
        self.scene.step(linear, angular, self.arm_delta, self.gripper)
        self.arm_delta = None
        self.steps += 1
        self.sim_time += self.control_timestep
        return True

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
                  running=time.monotonic() <= self.active_until, base=self.args.base, joint_names=self.joint_names,
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
            last_publish = 0.0
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
                if start - last_publish >= PUBLISH_PERIOD:
                    sim.snapshot()
                    last_publish = start
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
