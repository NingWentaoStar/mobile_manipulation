"""Top-down grasp of an object from a surface."""
import numpy as np

from ..manipulation.pose import Pose, pose_from_axes
from .base import Skill, SkillError
from .context import APPROACH, GRIPPER_CLOSED, GRIPPER_OPEN
from .registry import register

MAX_FINGER_SPAN = 0.075
LIFT = 0.12


def top_down_grasps(world, name):
    """Top-down grasp candidates (TCP poses) for a resting object, best first."""
    pts = world.object_points(name)
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    xy = pts[:, :2]
    centre = (xy.min(0) + xy.max(0)) / 2
    cov = np.cov((xy - xy.mean(0)).T)
    _, vecs = np.linalg.eigh(cov)
    candidates = []
    for k in range(2):
        axis = np.r_[vecs[:, k], 0.0]
        proj = xy @ vecs[:, k]
        other = xy @ vecs[:, 1 - k]
        width = proj.max() - proj.min()
        if width > MAX_FINGER_SPAN:
            continue
        mid = vecs[:, k] * (proj.max() + proj.min()) / 2 + vecs[:, 1 - k] * (other.max() + other.min()) / 2
        height = zmax - zmin
        z = max(zmax - 0.04, zmin + 0.5 * height, zmin + 0.012)
        for sign in (1, -1):
            candidates.append((width, pose_from_axes(np.r_[mid, z], [0, 0, -1], sign * axis)))
    candidates.sort(key=lambda c: c[0])
    return [pose for _, pose in candidates], (zmin, zmax), centre


@register
class Pick(Skill):
    name = "pick"
    applies_to = "窄边小于 7.5 cm、放在够得着的台面上的物体，从上方抓"
    not_for = "从侧面抓；抽屉和冰箱里的物品；台面靠墙一侧太远的物品"
    preconditions = "底盘停稳；手里没有东西"

    def label(self, object):
        return f"pick({object})"

    def execute(self, object):
        ctx, name = self.context, object
        if ctx.held:
            raise SkillError(f"already holding {ctx.held}")
        world = ctx.world
        if not top_down_grasps(ctx.world, name)[0]:
            raise SkillError(f"{name} is wider than the gripper in every direction")
        held = False
        for attempt in range(3):
            ctx.planner.update_obstacles()
            feasible = []
            for grasp in top_down_grasps(ctx.world, name)[0]:
                grasp_qs = ctx.planner.ik(grasp)
                if grasp_qs:
                    _, pre_q = ctx.pre_pose(grasp, grasp_qs[0], APPROACH)
                    if pre_q is not None:
                        feasible.append((grasp, grasp_qs[0], pre_q))
            if not feasible:
                if attempt == 0:
                    raise SkillError(f"{name} is out of the arm's reach from this base pose")
                break
            grasp, grasp_q, pre_q = feasible[attempt % len(feasible)]
            ctx.set_gripper(GRIPPER_OPEN, steps=4)
            ctx.approach(grasp, grasp_q, "grasp", pre_q)
            ctx.set_gripper(GRIPPER_CLOSED, steps=15)
            if world.is_grasping(name):
                held = True
                break
            ctx.set_gripper(GRIPPER_OPEN)
            ctx.back_off()
        if not held:
            raise SkillError(f"fingers closed but {name} is not held ({attempt + 1} grasps tried)")
        z0 = float(world.object_position(name)[2])
        ctx.line(Pose(grasp.p + [0, 0, LIFT], grasp.q), what="lift pose", slow=True)
        lifted = float(world.object_position(name)[2]) - z0
        if not world.is_grasping(name) or lifted < 0.05:
            raise SkillError(f"{name} slipped during the lift")
        ctx.held = name
        tcp = ctx.planner.tcp_pose()
        pts = world.object_points(name)
        lo, hi = pts.min(0), pts.max(0)
        box_centre = Pose((lo + hi) / 2)
        ctx.planner.attach_box(hi - lo + 0.01, tcp.inv() * box_centre)
        return dict(object=name, lifted=round(lifted, 3))
