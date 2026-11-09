"""Drive the mobile base to a target pose; the arm holds while the chassis moves."""
import math

import numpy as np

from .base import Skill, SkillError
from .registry import register

POSITION_TOLERANCE = 0.025
YAW_TOLERANCE = 0.035
MAX_STEPS = 1200


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


@register
class Navigate(Skill):
    name = "navigate"
    applies_to = "空旷地面上的目标位姿 (x, y, yaw)，yaw 可省略"
    not_for = "需要绕开家具的路径：只做直线反馈控制，没有避障"
    preconditions = "目标位姿可达；底盘能运动"

    def label(self, target, **_):
        return f"navigate({float(target[0]):.2f}, {float(target[1]):.2f})"

    def execute(self, target):
        """Point-to-point driving to `target` = (x, y) or (x, y, yaw), no obstacle avoidance."""
        ctx = self.context
        target = [float(v) for v in target]
        if len(target) not in (2, 3):
            raise SkillError("target must be (x, y) or (x, y, yaw)")
        x, y = target[:2]
        yaw_goal = target[2] if len(target) == 3 else None
        scene, hold = ctx.scene, ctx.planner.qpos()[:7]
        for _ in range(MAX_STEPS):
            position, quat = scene.pose(scene.chassis)
            yaw = yaw_of(quat)
            dx, dy = x - position[0], y - position[1]
            distance = math.hypot(dx, dy)
            yaw_error = None if yaw_goal is None else wrap(yaw_goal - yaw)
            if distance < POSITION_TOLERANCE and (yaw_error is None or abs(yaw_error) < YAW_TOLERANCE):
                for _ in range(20):
                    ctx.drive(0.0, 0.0, hold)
                return dict(position=[round(float(position[0]), 3), round(float(position[1]), 3),
                                      round(float(yaw), 3)],
                            distance_error=round(distance, 3),
                            yaw_error=round(float(yaw_error), 3) if yaw_error is not None else None)
            if distance < POSITION_TOLERANCE:
                linear, angular = 0.0, float(np.clip(yaw_error*1.8, -0.6, 0.6))
            else:
                heading = wrap(math.atan2(dy, dx) - yaw)
                linear = min(0.22, max(distance*0.9, 0.035)) if abs(heading) < 0.3 else 0.0
                angular = float(np.clip(heading*2.0, -0.6, 0.6))
            ctx.drive(linear, angular, hold)
        raise SkillError(f"did not reach ({x:.2f}, {y:.2f}) after {MAX_STEPS} steps")
