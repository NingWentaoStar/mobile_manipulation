"""MuJoCo physics and rendering of the home scene."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..geometry import BASE_CAMERA_SIZE, CAMERA_FOV_Y, HAND_CAMERA_SIZE, RADIUS, TRACK  # noqa: E402
from .kitchen import KitchenScene  # noqa: E402
from .model import ARM_JOINTS, CONTROL_SUBSTEPS, FINGER_JOINTS, REST_POSITIONS, TIMESTEP, build, yaw_quat  # noqa: E402

BASE_JOINTS = {"wheels": ["left_wheel_joint", "right_wheel_joint"], "planar": ["base_x_joint", "base_y_joint", "base_yaw_joint"]}
SETTLE_STEPS, SETTLE_DAMPING = 1500, 0.9
CAMERAS = {"base_camera": BASE_CAMERA_SIZE, "hand_camera": HAND_CAMERA_SIZE}


class MujocoScene:
    control_timestep = TIMESTEP * CONTROL_SUBSTEPS

    def __init__(self, base="wheels", start=None, viewer=False, extend=None):
        self.base = base
        self.kitchen = KitchenScene.load()
        self.start = tuple(float(v) for v in (start if start is not None else self.kitchen.start))
        self.model, self.spec = build(self.kitchen.add_to, base, self.start, extend)
        self.data = mujoco.MjData(self.model)
        self.joint_names = BASE_JOINTS[base] + ARM_JOINTS + FINGER_JOINTS
        self.qpos_index = [self.model.joint(n).qposadr[0] for n in self.joint_names]
        self.qvel_index = [self.model.joint(n).dofadr[0] for n in self.joint_names]
        self.arm_q = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.arm_ctrl = [self.model.actuator(n).id for n in ARM_JOINTS]
        self.base_ctrl = [self.model.actuator(n).id for n in BASE_JOINTS[base]]
        self.gripper_ctrl = self.model.actuator("gripper").id
        self.chassis = self.model.body("chassis_link").id
        self.root = self.model.body("chassis_link" if base == "wheels" else "world_root").id
        self.tcp = self.model.site("tcp").id
        self.objects = {name: self.model.body(name).id for name in self.kitchen.movable}
        self.articulations = {name: {joint: self.model.joint(f"{name}/{joint}").qposadr[0] for joint in joints}
                              for name, joints in self.kitchen.joints.items()}
        self.settled = None
        self.settled = self.settle()
        self.renderers = {name: mujoco.Renderer(self.model, height=h, width=w) for name, (w, h) in CAMERAS.items()}
        self.viewer = None
        if viewer:
            from mujoco import viewer as passive_viewer
            self.viewer = passive_viewer.launch_passive(self.model, self.data)

    def reset(self, start=None):
        """Restore the scene; `start` = (x, y, yaw) parks the robot somewhere else from now on."""
        model, data = self.model, self.data
        if start is not None:
            self.start = tuple(float(v) for v in start)
        root_z = model.body_pos[self.root][2]
        mujoco.mj_resetData(model, data)
        for name, value in REST_POSITIONS.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        if self.base == "wheels":
            address = model.joint("root").qposadr[0]
            data.qpos[address:address + 3] = [self.start[0], self.start[1], root_z]
            data.qpos[address + 3:address + 7] = yaw_quat(self.start[2])
        else:
            data.mocap_pos[0] = [self.start[0], self.start[1], root_z]
            data.mocap_quat[0] = yaw_quat(self.start[2])
        self.place_objects()
        data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q]
        data.ctrl[self.gripper_ctrl] = REST_POSITIONS["panda_finger_joint1"]
        mujoco.mj_forward(model, data)

    def settle(self):
        """Resting poses of the props, with the robot out of the way: some start overlapping the furniture, and
        undamped MuJoCo contacts would kick them away."""
        model, data = self.model, self.data
        self.reset()
        addresses = [model.joint(name).dofadr[0] for name in self.objects]
        dofs = [a + i for a in addresses for i in range(6)]
        robot = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == model.body_rootid[self.root]]
        contacts = model.geom_contype[robot].copy(), model.geom_conaffinity[robot].copy()
        model.geom_contype[robot], model.geom_conaffinity[robot] = 0, 0
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)
            data.qvel[dofs] *= SETTLE_DAMPING
        model.geom_contype[robot], model.geom_conaffinity[robot] = contacts
        return {name: tuple(data.qpos[a:a + n].copy() for a, n in ((model.joint(name).qposadr[0], 3), (model.joint(name).qposadr[0] + 3, 4)))
                for name in self.objects}

    def place_objects(self):
        model, data = self.model, self.data
        for name, (position, quat) in (self.settled or self.kitchen.movable).items():
            address = model.joint(name).qposadr[0]
            data.qpos[address:address + 3] = position
            data.qpos[address + 3:address + 7] = quat

    def base_targets(self, linear, angular):
        if self.base == "wheels":
            return [(linear - angular*TRACK/2) / RADIUS, (linear + angular*TRACK/2) / RADIUS]
        yaw = self.data.qpos[self.model.joint("base_yaw_joint").qposadr[0]]
        return [linear*np.cos(yaw), linear*np.sin(yaw), angular]

    def step(self, linear, angular, arm_delta, gripper):
        """One control step. arm_delta is relative to the measured joints; None holds the current target."""
        data = self.data
        if arm_delta is not None:
            data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q] + np.asarray(arm_delta)
        data.ctrl[self.gripper_ctrl] = gripper
        data.ctrl[self.base_ctrl] = self.base_targets(linear, angular)
        if self.viewer is None:
            mujoco.mj_step(self.model, data, CONTROL_SUBSTEPS)
        else:
            with self.viewer.lock():
                mujoco.mj_step(self.model, data, CONTROL_SUBSTEPS)
            self.viewer.sync()

    def pose(self, body):
        return self.data.xpos[body].copy(), self.data.xquat[body].copy()

    def tcp_pose(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp])
        return self.data.site_xpos[self.tcp].copy(), quat

    def handle_pose(self, articulation, joint):
        """World (position, rotation matrix) of the handle bar moved by `joint`, and its description."""
        handle = self.kitchen.handles[articulation][joint]
        body = self.model.body(f"{articulation}/{handle['link']}").id
        rotation, local = self.data.xmat[body].reshape(3, 3), np.zeros(9)
        mujoco.mju_quat2Mat(local, np.array(handle["quat"]))
        return self.data.xpos[body] + rotation @ handle["pos"], rotation @ local.reshape(3, 3), handle

    def joint_positions(self):
        return self.data.qpos[self.qpos_index].copy()

    def joint_velocities(self):
        return self.data.qvel[self.qvel_index].copy()

    def capture_rgb(self):
        images = {}
        for name, renderer in self.renderers.items():
            renderer.update_scene(self.data, camera=name)
            images[name] = renderer.render().copy()
        return images

    def camera_intrinsics(self):
        result = {}
        for name, (width, height) in CAMERAS.items():
            focal = height / (2*np.tan(CAMERA_FOV_Y/2))
            result[name] = [[focal, 0.0, width/2], [0.0, focal, height/2], [0.0, 0.0, 1.0]]
        return result

    def articulation_positions(self):
        return {name: {joint: float(self.data.qpos[address]) for joint, address in joints.items()}
                for name, joints in self.articulations.items()}

    def layout(self):
        return {**self.kitchen.layout, "start": list(self.start), "robot": f"mobile_panda_{self.base}"}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        for renderer in self.renderers.values():
            renderer.close()
