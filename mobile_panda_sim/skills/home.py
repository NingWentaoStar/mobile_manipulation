"""Bring the arm back to its rest pose."""
import numpy as np

from ..geometry import ARM_REST
from .base import Skill, SkillError
from .registry import register


@register
class Home(Skill):
    name = "home"
    applies_to = "任何位置，沿无碰撞路径回收手臂"
    not_for = "手臂被卡住时"
    preconditions = "底盘停稳"

    def execute(self):
        ctx = self.context
        ctx.planner.update_obstacles(exclude={ctx.held} if ctx.held else ())
        trajectory = ctx.planner.plan_home(np.asarray(ARM_REST[:7]))
        if trajectory is None:
            trajectory = ctx.planner.plan_home(np.asarray(ARM_REST[:7]), avoid=False)
            if trajectory is None:
                raise SkillError("rest pose unreachable")
        ctx.follow(trajectory)
