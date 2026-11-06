"""Put the held object down on a surface."""
import numpy as np

from ..manipulation.pose import Pose
from .base import Skill, SkillError
from .context import GRIPPER_OPEN
from .pick import LIFT
from .registry import register


@register
class Place(Skill):
    name = "place"
    applies_to = "平整台面"
    not_for = "放进抽屉或冰箱；倾斜或很小的表面；圆的物体松手后可能滚动几厘米"
    preconditions = "手里握着物体，底盘停稳"

    def execute(self, target, height=None):
        """Bottom of the object centred on target (x, y), on the surface below it or at `height`."""
        ctx = self.context
        if not ctx.held:
            raise SkillError("not holding anything")
        pts = ctx.world.object_points(ctx.held)
        bottom = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, pts[:, 2].min()]
        tcp = ctx.planner.tcp_pose()
        offset = tcp.p - bottom
        x, y = float(target[0]), float(target[1])
        support = height if height is not None else ctx.world.surface(x, y, exclude=ctx.held)
        if support is None:
            raise SkillError(f"no surface under ({x:.2f}, {y:.2f})")
        goal = Pose(np.r_[x, y, support + 0.012] + offset, tcp.q)
        ctx.planner.update_obstacles(exclude={ctx.held})
        goal_qs = ctx.planner.ik(goal)
        pre_q = ctx.pre_pose(goal, goal_qs[0], LIFT)[1] if goal_qs else None
        if pre_q is None:
            raise SkillError("place pose is out of the arm's reach from this base pose")
        ctx.approach(goal, goal_qs[0], "place", pre_q)
        ctx.set_gripper(GRIPPER_OPEN, steps=12)
        ctx.planner.detach()
        name, ctx.held = ctx.held, None
        ctx.line(Pose(goal.p + [0, 0, LIFT], goal.q), what="retreat pose")
        for _ in range(10):
            ctx.step(ctx.planner.qpos()[:7])
        pts = ctx.world.object_points(name)
        placed = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, pts[:, 2].min()]
        error = float(np.linalg.norm(placed[:2] - [x, y]))
        if error > 0.04 or abs(placed[2] - support) > 0.03:
            raise SkillError(f"{name} ended {error:.3f} m from the target (bottom z {placed[2]:.3f}, "
                             f"surface {support:.3f})")
        return dict(object=name, position_error=round(error, 3), support=round(float(support), 3))
