"""Running a trained policy on the robot: observations in the training frame, one step, the result."""
from functools import cached_property

import mujoco
import numpy as np
import torch
from torch import nn

from ...geometry import GRIPPER_RANGE, MAX_JOINT_DELTA
from ...manipulation.pose import Pose
from ..base import Skill, SkillError
from .tasks import ACTION_SIZE, DISTURBED, DOCK, MAX_STEPS, NEIGHBOURS, OBS_SIZE, TARGETS, body_box, body_geoms, is_grasped

LINK0_HEIGHT = 0.9196429
TRAINED_LINK0 = Pose((DOCK[0], DOCK[1], LINK0_HEIGHT), (np.cos(DOCK[2]/2), 0, 0, np.sin(DOCK[2]/2)))


def load_policy(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    actor = nn.Sequential(nn.Linear(OBS_SIZE, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                          nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, ACTION_SIZE))
    actor.load_state_dict({k[len("actor_mean."):]: v for k, v in state.items() if k.startswith("actor_mean.")})
    return actor.eval()


class PolicySkill(Skill):
    """Works from the docking pose DOCK. Observations are moved so that panda_link0 sits where it did in training."""
    policy_file = None

    @cached_property
    def policy(self):
        return load_policy(self.policy_file)

    @cached_property
    def boxes(self):
        return {}

    def select(self, name):
        if name not in TARGETS:
            raise SkillError(f"the policies were trained on {TARGETS}, not {name}")
        scene, model = self.context.scene, self.context.scene.model
        self.name_, self.body = name, model.body(name).id
        self.target_geoms = body_geoms(model, self.body)
        self.finger_geoms = [body_geoms(model, model.body(n).id) for n in ("panda_leftfinger", "panda_rightfinger")]
        self.finger_q = [model.joint(n).qposadr[0] for n in ("panda_finger_joint1", "panda_finger_joint2")]
        self.finger_v = [model.joint(n).dofadr[0] for n in ("panda_finger_joint1", "panda_finger_joint2")]
        self.arm_v = [model.joint(f"panda_joint{i}").dofadr[0] for i in range(1, 8)]
        self.link0 = model.body("panda_link0").id
        if name not in self.boxes:
            data = mujoco.MjData(model)
            data.qpos[:] = scene.data.qpos
            address = model.joint(name).qposadr[0]
            data.qpos[address:address + 3], data.qpos[address + 3:address + 7] = scene.settled[name]
            mujoco.mj_forward(model, data)
            self.boxes[name] = body_box(model, data, self.body)
        self.centre_offset, self.size = self.boxes[name]

    def grasped(self):
        scene = self.context.scene
        return is_grasped(scene.model, scene.data, self.target_geoms, self.finger_geoms, self.finger_q)

    def centre(self):
        data = self.context.scene.data
        return data.xpos[self.body] + data.xmat[self.body].reshape(3, 3) @ self.centre_offset

    def to_training(self):
        data = self.context.scene.data
        return (TRAINED_LINK0 * Pose(data.xpos[self.link0], data.xquat[self.link0]).inv()).to_transformation_matrix()

    def observation(self, goal):
        """`goal` is in world coordinates."""
        scene, data, frame = self.context.scene, self.context.scene.data, self.to_training()
        rotation, shift = frame[:3, :3], frame[:3, 3]
        turn = np.zeros(4)
        mujoco.mju_mat2Quat(turn, rotation.reshape(-1))
        centre, tcp = rotation @ self.centre() + shift, rotation @ data.site_xpos[scene.tcp] + shift
        goal = rotation @ np.asarray(goal) + shift
        quat, tcp_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, data.xquat[self.body])
        mujoco.mju_mat2Quat(tcp_quat, (rotation @ data.site_xmat[scene.tcp].reshape(3, 3)).reshape(-1))
        return np.concatenate([data.qpos[scene.arm_q], data.qpos[self.finger_q], data.qvel[self.arm_v], data.qvel[self.finger_v],
                               [float(self.grasped())], tcp, tcp_quat, goal, centre, quat, centre - tcp, goal - centre, self.size]).astype(np.float32)

    def act(self, goal):
        low, high = GRIPPER_RANGE
        with torch.no_grad():
            action = self.policy(torch.from_numpy(self.observation(goal))).clamp(-1, 1).numpy()
        self.context.scene.step(0.0, 0.0, action[:7]*MAX_JOINT_DELTA, low + (action[7] + 1)/2*(high - low))
        self.context.counted()

    def neighbours(self):
        scene = self.context.scene
        return {n: scene.data.xpos[scene.model.body(n).id].copy() for n in (*TARGETS, *NEIGHBOURS) if n != self.name_}

    def moved(self, before):
        now = self.neighbours()
        return {n: round(float(np.linalg.norm(now[n] - before[n])), 3) for n in now if np.linalg.norm(now[n] - before[n]) > DISTURBED}

    def finish(self, before, reached, goal):
        """Details of a run that ended, or SkillError if it did not reach the goal or moved another object."""
        moved = self.moved(before)
        if not reached:
            error = float(np.linalg.norm(self.centre() - goal))
            raise SkillError(f"{error:.3f} m from the goal after {MAX_STEPS} steps", dict(moved=moved, error=round(error, 3)))
        if moved:
            raise SkillError(f"moved {sorted(moved)}", dict(moved=moved))
        return dict(moved=moved)
