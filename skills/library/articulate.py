"""Open and close drawers and doors by their handle."""
import math

import numpy as np

from ..manipulation.pose import Pose, pose_from_axes
from .base import Skill, SkillError
from .context import CONTACT_JOINT_STEP, GRIPPER_CLOSED, GRIPPER_OPEN, densify
from .registry import register

HAND_CLEARANCE = 0.008
STAGES = 3


def _rotate(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle)
            + axis * (axis @ vector) * (1 - math.cos(angle)))


_PALM = ((-0.033, -0.105, -0.115), (0.033, 0.105, -0.058))
_FINGERS = {"open": ((-0.013, -0.052, -0.058), (0.013, 0.052, 0.012)),
            "closed": ((-0.013, -0.03, -0.058), (0.013, 0.03, 0.012))}


def _corners(boxes):
    return np.array([[x, y, z] for lo, hi in boxes for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


_HAND = {state: _corners((_PALM, fingers)) for state, fingers in _FINGERS.items()}


def _panel_clearance(tcp, centre, normal, gap, fingers="closed"):
    """Smallest distance from the hand to the panel plane behind a handle bar (negative: penetrating)."""
    pts = _HAND[fingers]
    corners = tcp.to_transformation_matrix() @ np.c_[pts, np.ones(len(pts))].T
    return float(((corners[:3].T - centre) @ normal).min() + gap)


def _rotation_pose(centre, axis, angle):
    """World transform rotating by `angle` about the line through `centre` along `axis`."""
    q = np.r_[math.cos(angle/2), np.asarray(axis) / np.linalg.norm(axis) * math.sin(angle/2)]
    centre = np.asarray(centre, float)
    return Pose(centre) * Pose(q=q) * Pose(-centre)


class DragPlanner:
    """Works out how to hold a link and which arm motion moves it along its joint."""

    def __init__(self, context):
        self.context = context

    def handle_frame(self, articulation, joint):
        """Handle bar centre, bar axis and outward normal (world), and the handle record."""
        pose, handle = self.context.world.handle(articulation, joint)
        m = pose.to_transformation_matrix()
        return pose.p, m[:3, 0], m[:3, 2], handle

    def handle_grasps(self, articulation, joint):
        """TCP grasp poses on a round handle bar: fingers close across the bar at its centre line,
        approaching into the panel, tilted about the bar axis by 0, +/-20, 40, 60 deg; long (vertical)
        bars are also held near either end."""
        centre, bar, normal, handle = self.handle_frame(articulation, joint)
        slack = max(handle.length / 2 - 0.04, 0.0)
        upward = bar if bar[2] >= 0 else -bar
        grasps = []
        for along in (0.0, 0.5, -0.5, 0.9, -0.9):
            point = centre + upward * along * slack
            for tilt in (0, 20, -20, 40, -40, 60, -60):
                approach = _rotate(-normal, bar, math.radians(tilt))
                closing = np.cross(bar, approach)
                for sign in (1, -1):
                    pose = pose_from_axes(point, approach, sign * closing)
                    if _panel_clearance(pose, centre, normal, handle.standoff, "open") >= 0.003:
                        grasps.append(pose)
            if slack == 0:
                break
        return grasps

    def joint_model(self, articulation, joint, target, opening):
        info = self.context.world.joint_info(articulation, joint)
        kind, q0, anchor, axis = info["kind"], info["position"], info["anchor"], info["axis"]
        lo, hi = info["limits"]
        if opening:
            goal = target if target is not None else (0.25 if kind == "prismatic" else 0.6)
        else:
            goal = 0.0
        goal = float(np.clip(goal, lo, hi))

        def moved(pose, dq):
            if kind == "prismatic":
                return Pose(pose.p + axis * dq, pose.q)
            return _rotation_pose(anchor, axis, dq) * pose
        return dict(kind=kind, q0=q0, goal=goal, moved=moved, names=(articulation, joint),
                    step=0.01 if kind == "prismatic" else 0.02, anchor=anchor, axis=axis)

    def panel(self, articulation, joint):
        """Door panel of a revolute joint: centre, normal (thin axis), vertical and radial (hinge -> free
        edge) axes, half extents along them, from the link's largest collision box."""
        m, half = self.context.world.panel_box(articulation, joint)
        model = self.joint_model(articulation, joint, None, True)
        hinge, axis = model["anchor"], model["axis"]
        order = np.argsort(half)
        normal_i = order[0]
        vertical_i = max((i for i in range(3) if i != normal_i), key=lambda i: abs(m[:3, i] @ axis))
        radial_i = 3 - normal_i - vertical_i
        radial = m[:3, radial_i] if m[:3, radial_i] @ (m[:3, 3] - hinge) > 0 else -m[:3, radial_i]
        hinge_along = (hinge - m[:3, 3]) @ radial
        return dict(centre=m[:3, 3], normal=m[:3, normal_i], vertical=m[:3, vertical_i], radial=radial,
                    half=(half[normal_i], half[vertical_i], half[radial_i]), hinge_along=hinge_along)

    def drag_candidates(self, articulation, joint, model=None, offset=0.0):
        """Ways to hold a link while moving it: fingers around the handle bar, approaching at several tilts
        about the bar and (long bars) heights along it; rolls about the bar (TCP x) while dragging.
        Candidates are dicts (grasp pose, roll axis, palm clearance, expected finger gap) so other
        holds, e.g. pushing a door face shut, can be added."""
        candidates = []
        centre, bar, normal, handle = self.handle_frame(articulation, joint)
        for grasp in self.handle_grasps(articulation, joint):
            candidates.append(dict(kind="handle", grasp=grasp, roll_axis=0, clearance=(centre, normal, handle.standoff),
                                   width=2 * handle.radius))
        if offset and model is not None:
            moved = model["moved"]
            for cand in candidates:
                cand["grasp"] = moved(cand["grasp"], offset)
                if cand["clearance"] is not None:
                    c, n_, gap = cand["clearance"]
                    tip = moved(Pose(c + n_), offset).p
                    c2 = moved(Pose(c), offset).p
                    cand["clearance"] = (c2, tip - c2, gap)
        return candidates

    def drag_path(self, model, cand, grasp, dqs, seed, max_jump=0.35, avoid=False, offset=0.0):
        """IK chain that moves a held link through joint offsets `dqs`, starting from TCP pose `grasp`. The
        gripper may roll about the candidate's roll axis (within +/-80 deg) to stay reachable. Returns
        (joint waypoints, steps reached)."""
        axis = grasp.to_transformation_matrix()[:3, cand["roll_axis"]]
        centre = grasp.p.copy()
        roll, path = 0.0, []
        min_clear = None if cand["clearance"] is None else \
            min(HAND_CLEARANCE, _panel_clearance(grasp, *cand["clearance"]) - 0.002)
        door = self.panel(*model["names"]) if model["kind"] == "revolute" else None
        if door is not None:
            _, _, handle_normal, _ = self.handle_frame(*model["names"])
            door["out"] = door["normal"] if door["normal"] @ handle_normal > 0 else -door["normal"]
        for dq in dqs:
            best = None
            for delta in (0.0, 0.1, -0.1, 0.2, -0.2, 0.35, -0.35):
                r = roll + delta
                if abs(r) > math.radians(80):
                    continue
                rolled = _rotation_pose(centre, axis, r) * grasp
                if min_clear is not None and _panel_clearance(rolled, *cand["clearance"]) < min_clear:
                    continue
                solutions = self.context.planner.ik(model["moved"](rolled, dq), seed=seed, n_init=4, avoid=avoid)
                if door is not None:
                    solutions = [q for q in solutions if self.outside_door(q, door, model, offset + dq)]
                if solutions:
                    jump = np.abs(solutions[0] - seed).max()
                    if jump <= max_jump and (best is None or jump < best[0]):
                        best = (jump, solutions[0], r)
                if best is not None and best[0] < 0.08:
                    break
            if best is None:
                break
            _, seed, roll = best
            path.append(seed)
        return path, len(path)

    def outside_door(self, q7, door, model, dq, margin=0.06):
        """Arm links 3-6 stay >= margin on the outer side of the door (at joint offset dq), or beyond its
        edges."""
        centre = model["moved"](Pose(door["centre"]), dq).p
        rot = model["moved"](Pose(), dq).to_transformation_matrix()[:3, :3]
        out, vertical, radial = rot @ door["out"], rot @ door["vertical"], rot @ door["radial"]
        half_n, half_v, half_r = door["half"]
        for p in self.context.planner.link_positions(q7)[:4]:
            d = p - centre
            if abs(d @ vertical) > half_v + margin or abs(d @ radial) > half_r + margin:
                continue
            if d @ out < half_n + margin:
                return False
        return True

    def drag_targets(self, model, q_now):
        """Joint offsets from q_now to the goal plus a small overshoot (the grasp gives a little; a closing
        link is pressed shut), one per drag step."""
        over = 0.02 if model["kind"] == "prismatic" else 0.05
        span = model["goal"] - q_now
        final = span + math.copysign(over, span)
        n = max(2, int(abs(final) / model["step"]))
        return np.linspace(0, final, n + 1)[1:], abs(span)

    def progress(self, model, cand, grasp_q, q_now, avoid=True, coarse=False, offset=0.0):
        """Joint distance (m or rad) towards the goal this candidate can drag. By default the same IK chain
        (step size, allowed joint jump) as the real drag, so the choice is not over-optimistic; `coarse`
        (base-pose search) checks every other step with a looser jump limit, ~3x faster."""
        dqs, span = self.drag_targets(model, q_now)
        if coarse and len(dqs) > 3:
            dqs = dqs[1::2]
        _, reached = self.drag_path(model, cand, cand["grasp"], dqs, grasp_q, max_jump=0.5 if coarse else 0.35,
                                     avoid=avoid, offset=offset)
        return min(abs(dqs[reached - 1]) if reached else 0.0, span)

    def best_drag(self, model, link, q_now, full_cloud, free_cloud, need=None, offset=0.0, skip=(), coarse=False):
        """Best reachable drag candidate from the current base pose: (progress, cand, grasp_q, pre_q).
        `offset` plans for the link `offset` further along than it is now (the full cloud, which has the
        link at its current pose, is then not used)."""
        best = None
        for i, cand in enumerate(self.drag_candidates(*model["names"], model, offset)):
            if i in skip:
                continue
            cand["index"] = i
            self.context.planner.use_cloud(free_cloud)
            solutions = self.context.planner.ik(cand["grasp"], avoid=True)
            if not solutions:
                continue
            self.context.planner.use_cloud(full_cloud if not offset else free_cloud)
            _, pre_q = self.context.pre_pose(cand["grasp"], solutions[0], None)
            if pre_q is None:
                continue
            self.context.planner.use_cloud(free_cloud)
            progress = self.progress(model, cand, solutions[0], q_now, coarse=coarse, offset=offset)
            if best is None or progress > best[0] + 1e-6:
                best = (progress, cand, solutions[0], pre_q)
            if progress >= (need if need is not None else abs(model["goal"] - q_now)) - 1e-6:
                break
        return best

    def plan_stages(self, model, full_cloud, free_cloud, start=None, stages=3):
        """Kinematic estimate (no physics) of the share of the way from joint position `start` (default: where
        it is now) to model["goal"] that up to `stages` grasp-and-drag stages cover from the current base pose."""
        q = model["q0"] if start is None else start
        span = abs(model["goal"] - q)
        sign = math.copysign(1.0, model["goal"] - q)
        total = 0.0
        for _ in range(stages):
            best = self.best_drag(model, None, q, full_cloud, free_cloud, need=span - total, offset=q - model["q0"],
                                   coarse=True)
            if best is None or best[0] < 0.01:
                break
            total += best[0]
            q += sign * best[0]
            if total >= span - 1e-6:
                break
        return total / max(span, 1e-6)


class Articulate(Skill):
    opening = True

    def label(self, articulation, joint, **_):
        return f"{self.name}({articulation}/{joint})"

    def actuate(self, articulation, joint, target, stages=STAGES):
        ctx, drag, opening = self.context, DragPlanner(self.context), self.opening
        if ctx.held:
            raise SkillError(f"holding {ctx.held}; place it first")
        model = drag.joint_model(articulation, joint, target, opening)
        kind, goal = model["kind"], model["goal"]
        if abs(goal - model["q0"]) < 0.01:
            return dict(joint_position=round(model["q0"], 3), note="already there")
        link = ctx.world.moving_link(articulation, joint)
        tolerance = 0.03 if kind == "prismatic" else 0.08
        sign = math.copysign(1.0, goal - model["q0"])
        short_of_goal = lambda q: (goal - q) * sign     # past the goal is fine: drags overshoot on purpose
        log, failed = [], set()
        ctx.settle_base()
        for stage in range(stages + 2):
            model = drag.joint_model(articulation, joint, target, opening)
            q_now = model["q0"]
            if short_of_goal(q_now) <= tolerance / 2:
                break
            full, free = ctx.planner.build_cloud(), ctx.planner.build_cloud(exclude={link})
            best = drag.best_drag(model, link, q_now, full, free, skip=failed)
            if best is None or best[0] < 0.01:
                if stage == 0:
                    raise SkillError(f"{articulation}/{joint} cannot be grasped and moved from this base pose "
                                     f"(see suggest_base_pose)")
                break
            _, cand, grasp_q, pre_q = best
            ctx.planner.use_cloud(full)
            ctx.set_gripper(GRIPPER_OPEN, steps=4)
            try:
                ctx.approach(cand["grasp"], grasp_q, "grasp", pre_q)
            except SkillError as error:
                log.append(dict(grasp=cand["kind"], held=False, approach=str(error)))
                failed.add(cand["index"])
                continue
            ctx.set_gripper(GRIPPER_CLOSED, steps=15)
            gap = float(ctx.planner.qpos()[7:].sum())
            if not ctx.world.link_grasped(articulation, joint) or abs(gap - cand["width"]) > 0.01:
                ctx.set_gripper(GRIPPER_OPEN)
                ctx.back_off()
                log.append(dict(grasp=cand["kind"], held=False, gap=round(gap, 3)))
                failed.add(cand["index"])
                continue
            model = drag.joint_model(articulation, joint, target, opening)
            dqs, _ = drag.drag_targets(model, model["q0"])
            ctx.planner.use_cloud(ctx.planner.build_cloud(exclude={link}, clear=(ctx.planner.tcp_pose().p, 0.08)))
            path, reached = drag.drag_path(model, cand, ctx.planner.tcp_pose(), dqs, ctx.planner.qpos()[:7], avoid=True)
            checked = True
            if reached < 3:
                path, reached = drag.drag_path(
                    model, cand, ctx.planner.tcp_pose(), dqs, ctx.planner.qpos()[:7], avoid=False)
                checked = False
            if path:
                ctx.follow(densify(np.vstack([ctx.planner.qpos()[:7], path]), CONTACT_JOINT_STEP), settle=15)
            ctx.set_gripper(GRIPPER_OPEN, steps=10)
            ctx.planner.use_cloud(full)
            ctx.back_off()
            q_after = ctx.world.joint_position(articulation, joint)
            log.append(dict(grasp=cand["kind"], held=True, start=round(model["q0"], 3), to=round(q_after, 3),
                            planned=f"{reached}/{len(dqs)}", predicted=round(best[0], 3), collision_checked=checked))
            if abs(q_after - model["q0"]) < 0.005:
                failed.add(cand["index"])
            if sum(1 for l in log if l["held"]) >= stages:
                break
        q_after = ctx.world.joint_position(articulation, joint)
        info = dict(joint_position=round(q_after, 3), goal=round(goal, 3), stages=log)
        if short_of_goal(q_after) > tolerance:
            raise SkillError(f"joint stopped at {q_after:.3f} (goal {goal:.3f}) after {len(log)} grasp(s); "
                             f"the rest is out of reach from this base pose", info)
        return info


@register
class Open(Articulate):
    name = "open"
    applies_to = "台柜上层左、中抽屉；冰箱上门到约 35°"
    not_for = "把手低于约 0.55 m 的抽屉和柜门；冰箱下门；矮柜推拉门"
    preconditions = "底盘停稳；手里没有东西"

    def execute(self, articulation, joint, target=None):
        """`target` is the joint position (m or rad); default 0.25 m for drawers, 0.6 rad for doors."""
        return self.actuate(articulation, joint, target)


@register
class Close(Articulate):
    name = "close"
    opening = False
    applies_to = "台柜上层左、中抽屉；冰箱上门从约 35° 关上"
    not_for = "把手低于约 0.55 m 的抽屉和柜门；从 70° 关冰箱门"
    preconditions = "底盘停稳；手里没有东西"

    def execute(self, articulation, joint):
        return self.actuate(articulation, joint, 0.0)
