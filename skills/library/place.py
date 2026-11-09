"""Put the held object down on a surface."""
import numpy as np

from ..manipulation.pose import Pose
from .base import Skill, SkillError
from .context import GRIPPER_OPEN
from .registry import register


@register
class Place(Skill):
    name = "place"
    applies_to = "平整台面"
    not_for = "放进抽屉或冰箱；倾斜或很小的表面；圆的物体松手后可能滚动几厘米"
    preconditions = "手里握着物体，底盘停稳"

    def execute(self, target, height=None, frame="base"):
        """Bottom of the object centred on target (x, y) in `frame`: 'base' = chassis_link (x forward,
        y left), 'map' = world. Height comes from the surface below, or from `height` (map frame)."""
        ctx = self.context
        if not ctx.held:
            raise SkillError("not holding anything")
        if frame not in ("base", "map"):
            raise SkillError("frame must be 'base' or 'map'")
        pts = ctx.world.object_points(ctx.held)
        bottom = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, pts[:, 2].min()]
        tcp = ctx.planner.tcp_pose()
        offset = tcp.p - bottom
        x, y = float(target[0]), float(target[1])
        if frame == "base":
            world = ctx.world.chassis_pose() * Pose([x, y, 0.0])
            x, y = float(world.p[0]), float(world.p[1])
        support = height if height is not None else ctx.world.surface(x, y, exclude=ctx.held)
        if support is None:
            raise SkillError(f"no surface under ({x:.2f}, {y:.2f})")
        goal = Pose(np.r_[x, y, support + 0.012] + offset, tcp.q)
        ctx.planner.update_obstacles(exclude={ctx.held})
        goal_q = pre_q = None
        for candidate in ctx.planner.ik(goal)[:8]:   # nearest joint branch first
            pre_q = ctx.pre_pose(goal, candidate, None)[1]   # try 0.10 / 0.06 / 0.04 m back-offs
            if pre_q is not None:
                goal_q = candidate
                break
        if pre_q is None:
            raise SkillError(f"place target ({x:.2f}, {y:.2f}) in map is out of the arm's reach "
                             f"from this base pose; move the base and retry")
        ctx.approach(goal, goal_q, "place", pre_q)
        ctx.set_gripper(GRIPPER_OPEN, steps=12)
        ctx.planner.detach()
        name, ctx.held = ctx.held, None
        ctx.back_off()   # best effort: pull straight up, trying 10/6/3 cm
        for _ in range(10):
            ctx.step(ctx.planner.qpos()[:7])
        pts = ctx.world.object_points(name)
        placed = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, pts[:, 2].min()]
        error = float(np.linalg.norm(placed[:2] - [x, y]))
        if error > 0.04 or abs(placed[2] - support) > 0.03:
            raise SkillError(f"{name} ended {error:.3f} m from the target (bottom z {placed[2]:.3f}, "
                             f"surface {support:.3f})")
        return dict(object=name, position_error=round(error, 3), support=round(float(support), 3),
                    target_map=[round(x, 3), round(y, 3)])
