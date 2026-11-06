"""Trained policy: set the held object down on the entry console, let go and lift the hand."""
import numpy as np

from ...paths import POLICIES
from ..base import SkillError
from ..registry import register
from .common import PolicySkill
from .tasks import AT_GOAL, HAND_ABOVE, MAX_STEPS, OPEN_GAP, TABLE_Z


@register
class Place(PolicySkill):
    name = "rl_place"
    policy_file = POLICIES / "place.pt"
    applies_to = "手中的柠檬、梨、汤罐，放到玄关台工作区内"
    not_for = "其他物体和其他台面"
    preconditions = "手里握着物体（来自 rl_pick_move），底盘停在 DOCK"

    def execute(self, target):
        """Set the held object down at target (x, y)."""
        ctx = self.context
        if not ctx.held:
            raise SkillError("not holding anything")
        self.select(ctx.held)
        goal = np.array([target[0], target[1], TABLE_Z + self.size[2]])
        before = self.neighbours()
        scene, data = ctx.scene, ctx.scene.data
        for _ in range(MAX_STEPS):
            self.act(goal)
            released = np.linalg.norm(self.centre() - goal) <= AT_GOAL and not self.grasped()
            open_hand = data.qpos[self.finger_q].sum() > OPEN_GAP
            clear = data.site_xpos[scene.tcp][2] - self.centre()[2] > HAND_ABOVE
            if released and open_hand and clear and np.linalg.norm(data.cvel[self.body]) < 0.1:
                info = self.finish(before, True, goal)
                ctx.held = None
                return info
        return self.finish(before, False, goal)
