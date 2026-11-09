"""Arm motion planning for the mobile Panda: IK, collision-free RRT and straight-line (screw) motion."""
import os
import time

import mplib
import numpy as np

from .pose import Pose

CLOUD_RADIUS = 1.25
CLOUD_DENSITY = 8000
IDLE_PERIOD = 1 / 30
CLOUD_POINT_RADIUS = 0.015
FAR_POINT = np.array([[50.0, 50.0, 50.0]])


def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


def planner_description():
    """Panda URDF/SRDF for mplib with the mobile base (pedestal + chassis boxes) fixed under panda_link0,
    so arm-vs-own-base collisions are self-collisions. Written to outputs/planner/."""
    from ..geometry import HEIGHT
    from ..paths import DESCRIPTION, OUTPUTS, PANDA_URDF
    pillar = HEIGHT - 0.27
    out = OUTPUTS / "planner"
    out.mkdir(parents=True, exist_ok=True)
    meshes = out / "franka_description"
    if meshes.is_symlink():
        meshes.unlink(missing_ok=True)
    meshes.mkdir(exist_ok=True)
    link, target = meshes / "meshes", DESCRIPTION / "meshes/franka"
    if link.is_symlink() and link.resolve() != target.resolve():   # stale after a move/rename
        link.unlink()
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    urdf = PANDA_URDF.read_text()
    pad = 0.01
    base = f"""
  <link name="mobile_base">
    <collision><origin xyz="0 0 {-pillar/2:.4f}"/><geometry><box size="{0.18+2*pad} {0.18+2*pad} {pillar:.4f}"/></geometry></collision>
    <collision><origin xyz="0 0 {-(pillar + 0.07):.4f}"/><geometry><box size="{0.55+2*pad} {0.46+2*pad} 0.14"/></geometry></collision>
  </link>
  <joint name="mobile_base_mount" type="fixed"><parent link="panda_link0"/><child link="mobile_base"/><origin xyz="0 0 0"/></joint>
</robot>"""
    urdf = urdf[:urdf.rindex("</robot>")] + base
    srdf = PANDA_URDF.with_suffix(".srdf").read_text()
    srdf = srdf[:srdf.rindex("</robot>")] + \
        '    <disable_collisions link1="panda_link0" link2="mobile_base" reason="Adjacent"/>\n</robot>'
    for name, text in (("mobile_panda_arm.urdf", urdf), ("mobile_panda_arm.srdf", srdf)):
        path = out / name
        if path.exists() and path.read_text() == text:
            continue
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    return out / "mobile_panda_arm.urdf"


class ArmPlanner:
    def __init__(self, world, joint_vel_limit=0.8, joint_acc_limit=1.5):
        self.world = world
        self.dt = world.control_timestep
        urdf = planner_description()
        self.planner = mplib.Planner(
            urdf=str(urdf), srdf=str(urdf.with_suffix(".srdf")), move_group="panda_hand_tcp",
            joint_vel_limits=np.full(7, joint_vel_limit), joint_acc_limits=np.full(7, joint_acc_limit))
        self.planner.set_base_pose(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64))
        self.cloud_ready = False
        self._active_cloud = None
        self.idle = None
        self._last_idle = 0.0

    def tick(self):
        """Give the caller a chance to run (e.g. keep a viewer window responsive) while planning."""
        if self.idle is not None:
            now = time.monotonic()
            if now - self._last_idle >= IDLE_PERIOD:
                self._last_idle = now
                self.idle()

    def base_pose(self):
        """World pose of panda_link0 (the planner's origin); `virtual_base` overrides it to evaluate
        reachability from a base pose the robot is not at yet."""
        if getattr(self, "virtual_base", None) is not None:
            return self.virtual_base
        return self.world.link0_pose()

    def link0_at(self, x, y, yaw):
        """panda_link0 world pose if the chassis were parked at (x, y, yaw)."""
        return Pose([x, y, self.world.link0_pose().p[2]], [np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

    def to_base(self, world_pose):
        return self.base_pose().inv() * world_pose

    def tcp_pose(self):
        return self.world.tcp_pose()

    def qpos(self):
        """Planner qpos: 7 arm joints + 2 fingers (mplib order)."""
        return self.world.qpos()

    @staticmethod
    def _vec(pose):
        return np.concatenate([pose.p, pose.q]).astype(np.float64)

    def update_obstacles(self, exclude=()):
        """Build the obstacle cloud (see build_cloud) and make it the active one. Returns its size."""
        self.use_cloud(self.build_cloud(exclude))
        return len(self._cloud)

    def use_cloud(self, cloud):
        """Make a cloud from build_cloud() the active obstacle set."""
        self._cloud = cloud if len(cloud) else FAR_POINT
        self.cloud_ready = True
        self._use_cloud(True)

    def build_cloud(self, exclude=(), clear=None, points=None):
        """Sample the scene's collision geometry near the arm into the planner's point cloud. `points` reuses
        scene_points() instead of sampling, for evaluating many base poses."""
        base = self.base_pose()
        inv = base.inv().to_transformation_matrix()
        if points is None:
            cloud = self.world.surface_points(exclude, base.p, CLOUD_RADIUS, CLOUD_DENSITY, self.tick)
        else:
            cloud = points[np.linalg.norm(points - base.p, axis=1) < CLOUD_RADIUS]
        if clear is not None:
            cloud = cloud[np.linalg.norm(cloud - np.asarray(clear[0]), axis=1) > clear[1]]
        cloud = cloud @ inv[:3, :3].T + inv[:3, 3]
        return cloud.astype(np.float64)

    def scene_points(self, exclude, centre, reach):
        """World points of the scene around `centre`, enough for any arm base within `reach` of it."""
        return self.world.surface_points(exclude, np.asarray(centre), CLOUD_RADIUS + reach, CLOUD_DENSITY, self.tick)

    def attach_box(self, size, tcp_to_box):
        """Treat a held object as a box (full size, pose relative to the TCP) during planning."""
        self.planner.update_attached_box(np.asarray(size, dtype=np.float64), self._vec(tcp_to_box))
        self.attached = True

    def detach(self):
        self.attached = False

    def _use_cloud(self, enabled):
        """mplib's IK rejects solutions touching the cloud, so contact poses swap in a far-away point. Rebuilding
        the collision tree is the slowest step of planning: do it only when the cloud changes."""
        cloud = self._cloud if enabled and self.cloud_ready else FAR_POINT
        if cloud is not self._active_cloud:
            self.planner.update_point_cloud(cloud, radius=CLOUD_POINT_RADIUS)
            self._active_cloud = cloud

    def ik(self, world_pose, seed=None, n_init=30, avoid=False):
        """Arm joint solutions (7,) for a TCP world pose, closest to `seed` (default: current joints)
        first; [] if unreachable. Self-collision is always checked; obstacles only when `avoid`."""
        self.tick()
        start = self.qpos() if seed is None else np.concatenate([seed, self.qpos()[7:]])
        self._use_cloud(avoid)
        status, solutions = self.planner.IK(self._vec(self.to_base(world_pose)), start, n_init_qpos=n_init)
        if status != "Success" or solutions is None:
            return []
        solutions = [np.asarray(s)[:7] for s in solutions]
        return sorted(solutions, key=lambda s: np.abs(s - start[:7]).sum())

    def in_collision(self, q7, avoid=True):
        """True if arm joints q7 collide with itself or (when avoid) the obstacle cloud."""
        q = np.concatenate([q7, self.qpos()[7:]])
        if self.planner.check_for_self_collision(self.planner.robot, q):
            return True
        self._use_cloud(True)
        return bool(avoid and self.cloud_ready and self.planner.check_for_env_collision(
            self.planner.robot, q, with_point_cloud=True, use_attach=getattr(self, "attached", False)))

    def plan_to_pose(self, world_pose, avoid=True, planning_time=2.0):
        """Collision-free joint trajectory (N, 7) to a TCP world pose, or None. IK goals are computed and
        collision-filtered here (closest to the current joints first), then RRTConnect plans to them."""
        goals = [q for q in self.ik(world_pose, n_init=40, avoid=avoid) if not self.in_collision(q, avoid)]
        if not goals:
            return None
        return self.plan_joints(goals[:6], avoid, planning_time)

    def plan_joints(self, goals, avoid=True, planning_time=2.0):
        """Collision-free joint trajectory to any of the arm joint goals (list of (7,)), or None."""
        fingers = self.qpos()[7:]
        current = self.qpos()[:7]
        for g in goals:
            if np.abs(np.asarray(g) - current).max() < 1e-3:
                return np.asarray([current])
        goals = [np.concatenate([g, fingers]) for g in goals]
        self._use_cloud(avoid)
        for _ in range(3):
            try:
                result = self.planner.plan_qpos_to_qpos(goals, self.qpos(), time_step=self.dt, planning_time=planning_time,
                                                        use_point_cloud=avoid and self.cloud_ready,
                                                        use_attach=getattr(self, "attached", False) and avoid)
            except RuntimeError:
                continue
            if result["status"] == "Success":
                return np.asarray(result["position"])[:, :7]
        return None

    def fk(self, q7):
        """TCP world pose for arm joints q7 (fingers at their current opening)."""
        model = self.planner.robot.get_pinocchio_model()
        model.compute_forward_kinematics(np.concatenate([q7, self.qpos()[7:]]))
        pose = model.get_link_pose(self.planner.move_group_link_id)
        return self.base_pose() * Pose(pose[:3], pose[3:])

    def link_positions(self, q7, links=("panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7")):
        """World positions of arm link origins for joints q7 (for cheap clearance checks)."""
        model = self.planner.robot.get_pinocchio_model()
        model.compute_forward_kinematics(np.concatenate([q7, self.qpos()[7:]]))
        base = self.base_pose().to_transformation_matrix()
        out = []
        for name in links:
            p = model.get_link_pose(self.planner.link_name_2_idx[name])[:3]
            out.append(base[:3, :3] @ p + base[:3, 3])
        return np.asarray(out)

    def plan_home(self, q7, avoid=True, planning_time=2.0):
        """Collision-free joint trajectory to arm joints q7, or None."""
        return self.plan_joints([np.asarray(q7, float)], avoid, planning_time)

    def plan_line(self, world_pose, step=0.01, max_jump=0.25):
        """Straight-line TCP motion to a world pose (position interpolated, orientation slerped), as a
        timed joint trajectory, or None if any waypoint is unreachable. IK is chained from the current
        joints so the arm does not flip configuration. No obstacle check: used for short approaches
        where the gripper is meant to touch things."""
        start = self.tcp_pose()
        distance = np.linalg.norm(world_pose.p - start.p)
        angle = 2 * np.arccos(min(1.0, abs(float(np.dot(start.q, world_pose.q)))))
        n = max(2, int(max(distance / step, angle / 0.05)) + 1)
        seed, path = self.qpos()[:7], [self.qpos()[:7]]
        for t in np.linspace(0, 1, n)[1:]:
            pose = Pose(start.p + (world_pose.p - start.p) * t, _slerp(start.q, world_pose.q, t))
            solutions = self.ik(pose, seed=seed, n_init=6)
            if not solutions or np.abs(solutions[0] - seed).max() > max_jump:
                return None
            seed = solutions[0]
            path.append(seed)
        return self.time_path(np.asarray(path))

    def time_path(self, waypoints):
        """Time-parameterize joint waypoints (M, 7) into a trajectory sampled at the control rate."""
        path = np.asarray(waypoints, dtype=np.float64)
        if len(path) < 2:
            return path
        times, positions, *_ = self.planner.TOPP(path, self.dt)
        return np.asarray(positions)[:, :7]
