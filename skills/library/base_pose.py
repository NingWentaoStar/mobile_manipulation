"""Choose where to park the chassis for a drawer or a door."""
import math

import numpy as np

from ..geometry import BASE_HALF
from ..manipulation.pose import Pose
from .articulate import DragPlanner
from .base import Skill, SkillError
from .registry import register

SEARCH_REACH = 1.2
BAND_WIDTH = 0.1


def chassis_blocked(xy, yaw, footprints, margin=0.05):
    c, s_ = math.cos(yaw), math.sin(yaw)
    hx, hy = BASE_HALF[0] + margin, BASE_HALF[1] + margin
    corners = np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]]) @ np.array([[c, s_], [-s_, c]]) + xy
    lo, hi = corners.min(0), corners.max(0)
    return any(lo[0] < f[1] and hi[0] > f[0] and lo[1] < f[3] and hi[1] > f[2] for f in footprints)


@register
class SuggestBasePose(Skill):
    name = "suggest_base_pose"
    applies_to = "抽屉和柜门；不移动机器人"
    not_for = "不考虑底盘受力，假设底盘不动"
    preconditions = "无"

    def label(self, articulation, joint, **_):
        return f"suggest_base_pose({articulation}/{joint})"

    def execute(self, articulation, joint, target=None, round_trip=True):
        """info["pose"] = (x, y, yaw) from which the joint reaches `target` and, with `round_trip`, closes again;
        info["coverage"] is the reachable share of the worse direction. Takes the middle of the feasible distance
        band, not its edge."""
        ctx, drag = self.context, DragPlanner(self.context)
        model = drag.joint_model(articulation, joint, target, True)
        link_name = ctx.world.moving_link(articulation, joint)
        handle, _, normal, _ = drag.handle_frame(articulation, joint)
        if model["kind"] == "revolute":
            handle = model["moved"](Pose(handle), (model["goal"] - model["q0"]) / 2).p
        out = np.r_[normal[:2], 0.0]; out /= np.linalg.norm(out)
        side = np.array([-out[1], out[0], 0.0])
        blocked = [f["footprint"] for f in ctx.world.layout()["furniture"]]
        over = 0.02 if model["kind"] == "prismatic" else 0.05
        legs = [(model["q0"], model["goal"])]
        if round_trip:
            legs.append((model["goal"] + math.copysign(over, model["goal"] - model["q0"]), 0.0))
        full_points = ctx.planner.scene_points((), handle, SEARCH_REACH)
        free_points = ctx.planner.scene_points({link_name}, handle, SEARCH_REACH)
        candidates, first_full = [], None
        try:
            for distance in np.arange(0.40, 1.05, 0.025):
                if first_full is not None and distance > first_full + BAND_WIDTH:
                    break
                for lateral in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.45, 0.45, -0.6, 0.6):
                    xy = handle[:2] + out[:2] * distance + side[:2] * lateral
                    yaw = math.atan2(handle[1] - xy[1], handle[0] - xy[0])
                    if chassis_blocked(xy, yaw, blocked):
                        continue
                    ctx.planner.virtual_base = ctx.planner.link0_at(xy[0], xy[1], yaw)
                    if not any(ctx.planner.ik(c["grasp"], n_init=6) for c in
                               drag.drag_candidates(articulation, joint)[:40]):
                        continue
                    full = ctx.planner.build_cloud(points=full_points)
                    free = ctx.planner.build_cloud(points=free_points)
                    coverage = 1.0
                    for start, goal in legs:
                        coverage = min(coverage, drag.plan_stages(dict(model, goal=goal), full, free, start))
                        if coverage == 0:
                            break
                    if coverage > 0:
                        candidates.append((min(coverage, 1.0), float(distance), abs(lateral), (float(xy[0]), float(xy[1]), float(yaw))))
                    if coverage >= 0.999 and first_full is None:
                        first_full = float(distance)
        finally:
            ctx.planner.virtual_base = None
            ctx.planner.cloud_ready = False
        if not candidates:
            raise SkillError(f"no base pose reaches {articulation}/{joint}", dict(pose=None, coverage=0.0))
        best = max(c[0] for c in candidates)
        top = [c for c in candidates if c[0] >= best - 1e-3]
        distances = sorted(c[1] for c in top)
        middle = distances[len(distances) // 2]
        chosen = min(top, key=lambda c: (abs(c[1] - middle), c[2]))
        return dict(pose=chosen[3], coverage=chosen[0])
