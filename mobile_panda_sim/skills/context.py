"""What the skills share: the simulation, the planner, the gripper and what is held, and the basic arm motions."""
from functools import cached_property

import numpy as np

from ..geometry import MAX_JOINT_DELTA
from ..manipulation.planner import ArmPlanner
from ..manipulation.pose import Pose
from ..mj.world import MujocoWorld
from .base import SkillError

GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0
APPROACH = 0.10
CONTACT_JOINT_STEP = 0.015


def densify(path, max_step):
    """Linearly interpolate joint waypoints so no joint moves more than max_step per control step."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(np.ceil(np.abs(b - a).max() / max_step)))
        out.extend(a + (b - a) * (i / k) for i in range(1, k + 1))
    return np.asarray(out)


class SkillContext:
    def __init__(self, scene, on_step=None, on_idle=None):
        """on_step: called after every simulation step. on_idle: called ~30 times a second while planning."""
        self.scene = scene
        self.world = MujocoWorld(scene)
        self.on_step = on_step
        self.on_idle = on_idle
        self.reset()

    def reset(self):
        self.gripper = GRIPPER_OPEN
        self.held = None
        self.steps = 0
        if "planner" in self.__dict__:
            self.planner.detach()
            self.planner.cloud_ready = False

    @cached_property
    def planner(self):
        planner = ArmPlanner(self.world)
        planner.idle = self.on_idle
        return planner

    def counted(self):
        """Book one simulation step taken outside step(), by a policy."""
        self.steps += 1
        if self.on_step:
            self.on_step()

    def step(self, q_target=None):
        q = self.planner.qpos()[:7]
        delta = np.zeros(7) if q_target is None else np.clip(q_target - q, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
        self.world.step(delta, self.gripper)
        self.steps += 1
        if self.on_step:
            self.on_step()

    def follow(self, trajectory, tol=0.004, settle=40):
        for q in trajectory:
            self.step(q)
        target = trajectory[-1]
        for _ in range(settle):
            if np.abs(self.planner.qpos()[:7] - target).max() < tol:
                break
            self.step(target)

    def settle_base(self, steps=40, still=0.0003):
        """Hold the arm until the chassis stops moving; plans are made from its pose."""
        hold = self.planner.qpos()[:7]
        last = self.world.link0_pose().p
        for _ in range(steps):
            self.step(hold)
            now = self.world.link0_pose().p
            if np.linalg.norm(now - last) < still:
                break
            last = now

    def correct(self, grasp, tolerance=0.002, tries=2):
        """Remove the hand error left by base motion during an approach."""
        for _ in range(tries):
            if np.linalg.norm(self.planner.tcp_pose().p - grasp.p) < tolerance:
                return
            line = self.planner.plan_line(grasp, step=0.003)
            if line is None:
                return
            self.follow(densify(line, CONTACT_JOINT_STEP), tol=0.002)

    def set_gripper(self, value, steps=12):
        self.gripper = value
        hold = self.planner.qpos()[:7]
        for _ in range(steps):
            self.step(hold)

    def pre_pose(self, grasp, grasp_q, back_off):
        """Collision-free pre-grasp joints `back_off` m behind `grasp` along the approach, in the same IK
        branch as the grasp solution; tries shorter back-offs when the long one collides."""
        approach = grasp.to_transformation_matrix()[:3, 2]
        for distance in (APPROACH, 0.06, 0.04) if back_off is None else (back_off,):
            pre = Pose(grasp.p - approach * distance, grasp.q)
            for q in self.planner.ik(pre, seed=grasp_q, n_init=10, avoid=True):
                if np.abs(q - grasp_q).max() <= 0.6 and not self.planner.in_collision(q):
                    return pre, q
        return None, None

    def approach(self, grasp, grasp_q, what, pre_q):
        """Collision-free move to the pre-grasp joints, then a straight line to `grasp`."""
        trajectory = self.planner.plan_joints([pre_q])
        if trajectory is None:
            raise SkillError(f"no collision-free path to the pre-{what} pose")
        self.follow(trajectory)
        line = self.planner.plan_line(grasp)
        if line is None:
            line = densify(np.stack([self.planner.qpos()[:7], grasp_q]), 0.02)
        self.follow(densify(line, CONTACT_JOINT_STEP), tol=0.002)
        if np.abs(self.planner.qpos()[:7] - grasp_q).max() > 0.02 and \
                np.linalg.norm(self.planner.fk(grasp_q).p - grasp.p) < 0.005:
            self.follow(densify(np.stack([self.planner.qpos()[:7], grasp_q]), CONTACT_JOINT_STEP))
        self.correct(grasp)

    def back_off(self):
        """Best effort: pull the open gripper straight back along its approach axis (10, 6 or 3 cm)."""
        tcp = self.planner.tcp_pose()
        approach = tcp.to_transformation_matrix()[:3, 2]
        for distance in (APPROACH, 0.06, 0.03):
            line = self.planner.plan_line(Pose(tcp.p - approach * distance, tcp.q))
            if line is not None:
                self.follow(line)
                return True
        return False

    def line(self, pose, what="target", slow=False):
        trajectory = self.planner.plan_line(pose)
        if trajectory is None:
            raise SkillError(f"straight-line motion to the {what} is not reachable")
        self.follow(densify(trajectory, CONTACT_JOINT_STEP) if slow else trajectory)
