"""Trained policy: take an object from the entry console and hold it at a point."""
import numpy as np

from ...paths import POLICIES
from ..registry import register
from .common import PolicySkill
from .tasks import AT_GOAL, MAX_STEPS, STILL


@register
class PickMove(PolicySkill):
    name = "rl_pick_move"
    policy_file = POLICIES / "pick_move.pt"
    applies_to = "玄关台上的柠檬、梨、汤罐"
    not_for = "其他物体、其他台面、其他停靠位"
    preconditions = "底盘停在 DOCK；物体在玄关台工作区内"

    def label(self, object, **_):
        return f"rl_pick_move({object})"

    def execute(self, object, goal):
        """Hold `object` still with its centre at the world point `goal`."""
        self.select(object)
        goal = np.asarray(goal, float)
        before = self.neighbours()
        data = self.context.scene.data
        for _ in range(MAX_STEPS):
            self.act(goal)
            if np.linalg.norm(self.centre() - goal) <= AT_GOAL and np.linalg.norm(data.qvel[self.arm_v]) < STILL and self.grasped():
                info = self.finish(before, True, goal)
                self.context.held = object
                return info
        return self.finish(before, False, goal)
