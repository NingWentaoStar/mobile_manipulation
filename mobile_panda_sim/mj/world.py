"""What the skills need from a simulation, implemented on a MujocoScene."""
from types import SimpleNamespace

import mujoco
import numpy as np

from ..geometry import GRIPPER_RANGE
from ..manipulation.pose import Pose

FINGERS = ("panda_leftfinger", "panda_rightfinger")
GRASP_FORCE = 0.5
RAY_GROUPS = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
GROUND_HALF = 8.0


def sample_triangles(triangles, count, rng):
    """`count` points on triangles (T, 3, 3), area-weighted."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    picks = rng.choice(len(triangles), count, p=areas / areas.sum())
    u, v = rng.random((2, count))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[picks] + (b[picks] - a[picks])*u[:, None] + (c[picks] - a[picks])*v[:, None]


def box_triangles(half):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * half
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return corners[np.array(faces)]


class MujocoWorld:
    def __init__(self, scene):
        self.scene = scene
        self.model, self.data = scene.model, scene.data
        model = self.model
        self.control_timestep = scene.control_timestep
        self.link0 = model.body("panda_link0").id
        self.robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == model.body_rootid[scene.root]}
        self.fingers = [model.body(name).id for name in FINGERS]
        self.collision_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
        self.scene_geoms = [g for g in self.collision_geoms if model.geom_bodyid[g] not in self.robot_bodies]
        self.samples = {}

    def link0_pose(self):
        return Pose(self.data.xpos[self.link0], self.data.xquat[self.link0])

    def tcp_pose(self):
        return Pose(*self.scene.tcp_pose())

    def qpos(self):
        """7 arm joints + 2 fingers."""
        return self.scene.joint_positions()[-9:]

    def step(self, delta, gripper):
        low, high = GRIPPER_RANGE
        self.scene.step(0.0, 0.0, delta, low + (gripper + 1)/2*(high - low))

    def geom_triangles(self, geom):
        model = self.model
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, faces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            return vertices[model.mesh_face[start:start + faces]].astype(np.float64)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return box_triangles(model.geom_size[geom])
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            return box_triangles(np.array([GROUND_HALF, GROUND_HALF, 1e-4]))
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius, half = model.geom_size[geom][:2]
            return box_triangles(np.array([radius, radius, half + radius]))
        raise ValueError(f"unsupported collision geom type {kind}")

    def geom_samples(self, geom, density):
        """Surface points of a geom in its own frame; fixed, so each scene update only transforms them."""
        if geom not in self.samples:
            triangles = self.geom_triangles(geom)
            area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
            count = int(min(max(area*density, 50), 200000))
            self.samples[geom] = sample_triangles(triangles, count, np.random.default_rng(geom))
        return self.samples[geom]

    def surface_points(self, exclude, origin, radius, density, tick):
        """World points sampled from the scene's collision surfaces within `radius` of `origin`."""
        model, data = self.model, self.data
        skipped = {model.body(name).id for name in exclude}
        points = []
        for geom in self.scene_geoms:
            tick()
            if model.geom_bodyid[geom] in skipped:
                continue
            kind = model.geom_type[geom]
            if kind != mujoco.mjtGeom.mjGEOM_PLANE and np.linalg.norm(data.geom_xpos[geom] - origin) - model.geom_rbound[geom] > radius:
                continue
            world = data.geom_xpos[geom] + self.geom_samples(geom, density) @ data.geom_xmat[geom].reshape(3, 3).T
            points.append(world[np.linalg.norm(world - origin, axis=1) < radius])
        return np.concatenate(points) if points else np.zeros((0, 3))

    def object_points(self, name):
        model, data = self.model, self.data
        body = model.body(name).id
        points = []
        for geom in self.collision_geoms:
            if model.geom_bodyid[geom] == body:
                vertices = self.geom_triangles(geom).reshape(-1, 3)
                points.append(data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T)
        return np.concatenate(points)

    def object_position(self, name):
        return self.data.xpos[self.model.body(name).id].copy()

    def object_positions(self):
        return {name: self.data.xpos[body].copy() for name, body in self.scene.objects.items()}

    def fingers_press(self, body):
        """True if both fingers push on `body`."""
        model, data = self.model, self.data
        force = {finger: 0.0 for finger in self.fingers}
        wrench = np.zeros(6)
        for i in range(data.ncon):
            pair = model.geom_bodyid[data.contact.geom1[i]], model.geom_bodyid[data.contact.geom2[i]]
            if body in pair:
                other = pair[0] if pair[1] == body else pair[1]
                if other in force:
                    mujoco.mj_contactForce(model, data, i, wrench)
                    force[other] += wrench[0]
        return all(value > GRASP_FORCE for value in force.values())

    def is_grasping(self, name):
        return self.fingers_press(self.model.body(name).id)

    def link_grasped(self, articulation, joint):
        return self.fingers_press(self.model.body(self.moving_link(articulation, joint)).id)

    def surface(self, x, y, exclude=None, top=2.0):
        """Height of the first surface under (x, y), ignoring the robot and object `exclude`."""
        model = self.model
        skipped = set(self.robot_bodies)
        if exclude is not None:
            skipped.add(model.body(exclude).id)
        z, geom = top, np.zeros(1, dtype=np.int32)
        for _ in range(400):
            distance = mujoco.mj_ray(model, self.data, np.array([x, y, z]), np.array([0.0, 0.0, -1.0]), RAY_GROUPS, 1, -1, geom)
            if distance < 0:
                return None
            hit = z - distance
            if model.geom_bodyid[geom[0]] not in skipped and (model.geom_contype[geom[0]] or model.geom_conaffinity[geom[0]]):
                return float(hit)
            z = hit - 0.002
        return None

    def joint_id(self, articulation, joint):
        return self.model.joint(f"{articulation}/{joint}").id

    def joint_info(self, articulation, joint):
        model, data = self.model, self.data
        index = self.joint_id(articulation, joint)
        kind = "prismatic" if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_SLIDE else "revolute"
        return dict(kind=kind, limits=tuple(float(v) for v in model.jnt_range[index]), position=self.joint_position(articulation, joint),
                    anchor=data.xanchor[index].copy(), axis=data.xaxis[index].copy())

    def joint_position(self, articulation, joint):
        return float(self.data.qpos[self.scene.articulations[articulation][joint]])

    def moving_link(self, articulation, joint):
        """Body name of the link `joint` moves, as surface_points() takes it in `exclude`."""
        return self.model.body(self.model.jnt_bodyid[self.joint_id(articulation, joint)]).name

    def handle(self, articulation, joint):
        position, rotation, handle = self.scene.handle_pose(articulation, joint)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, position
        return Pose(matrix), SimpleNamespace(length=handle["length"], radius=handle["radius"], standoff=handle["standoff"])

    def panel_box(self, articulation, joint):
        """(4x4 world transform, half sizes) of the largest collision box on the link `joint` moves."""
        model, data = self.model, self.data
        body = model.jnt_bodyid[self.joint_id(articulation, joint)]
        boxes = [g for g in self.collision_geoms if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        box = max(boxes, key=lambda g: np.prod(model.geom_size[g]))
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = data.geom_xmat[box].reshape(3, 3), data.geom_xpos[box]
        return matrix, model.geom_size[box].astype(np.float64).copy()

    def layout(self):
        return self.scene.layout()
