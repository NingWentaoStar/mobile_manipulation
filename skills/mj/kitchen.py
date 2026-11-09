"""The home scene (kitchen, dining room, living room), built from assets/mujoco_kitchen."""
from dataclasses import dataclass, field
import json
import math

import mujoco
import numpy as np

from ..paths import KITCHEN

STATIC, ARTICULATED = (1, 1), (2, 1)   # (contype, conaffinity); articulated links skip each other
CAPSULE_X_TO_Z = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
SHAPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "mesh": mujoco.mjtGeom.mjGEOM_MESH}
LIGHTS = ((-1.6, 1.6), (1.4, 0.3), (2.4, -1.8), (-2.8, -1.4))
PROP_DAMPING = 0.05
# Without damping and sleeping, small props keep rocking and creep across the table.
SLEEP_TOLERANCE = 0.01


def transform(pos, quat):
    matrix = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, float))
    matrix[:3, :3], matrix[:3, 3] = rotation.reshape(3, 3), pos
    return matrix


def relative_pose(parent, child):
    """Pose of `child` in the frame of `parent`; both are dicts with world pos and quat."""
    matrix = np.linalg.inv(transform(parent["pos"], parent["quat"])) @ transform(child["pos"], child["quat"])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, matrix[:3, :3].reshape(-1))
    return matrix[:3, 3], quat


@dataclass
class KitchenScene:
    """Scene description the simulation needs at runtime."""
    data: dict
    meshes: object = None
    added: set = field(default_factory=set)

    @classmethod
    def load(cls):
        return cls(json.loads((KITCHEN / "scene.json").read_text(encoding="utf-8")), np.load(KITCHEN / "meshes.npz"))

    @property
    def start(self):
        return tuple(self.data["start"])

    @property
    def layout(self):
        return self.data["layout"]

    @property
    def movable(self):
        return {a["name"]: (a["pos"], a["quat"]) for a in self.data["actors"] if a["movable"]}

    @property
    def handles(self):
        return {a["name"]: a["handles"] for a in self.data["articulations"]}

    @property
    def joints(self):
        return {a["name"]: [link["joint"]["name"] for link in a["links"] if "joint" in link] for a in self.data["articulations"]}

    def add_mesh(self, spec, key):
        if key not in self.added:
            self.added.add(key)
            arrays = {name: self.meshes[f"{key}/{name}"] for name in ("v", "f", "n", "uv") if f"{key}/{name}" in self.meshes.files}
            mesh = spec.add_mesh(name=key, uservert=arrays["v"].reshape(-1).tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            if "f" in arrays:
                mesh.userface = arrays["f"].reshape(-1).tolist()
                mesh.usernormal = arrays["n"].reshape(-1).tolist()
                mesh.usertexcoord = arrays["uv"].reshape(-1).tolist()
                mesh.userfacenormal = mesh.userface
                mesh.userfacetexcoord = mesh.userface
        return key

    def add_material(self, spec, item):
        texture = item.get("texture")
        rgba = item["rgba"]
        name = f"m_{texture}_{hash(tuple(rgba)) & 0xffffffff:x}"
        if name not in self.added:
            self.added.add(name)
            material = spec.add_material(name=name, rgba=rgba, specular=0.2, shininess=0.1)
            if texture is not None:
                if texture not in self.added:
                    self.added.add(texture)
                    spec.add_texture(name=f"t_{texture}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                     file=str(KITCHEN / "textures" / f"{texture}.png"))
                material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"t_{texture}"
        return name

    def add_shape(self, spec, body, shape):
        """One geom; returns it unconfigured for contact, disabled by default."""
        kind = shape["type"]
        quat = np.array(shape["quat"], float)
        extra = {}
        if kind == "box":
            extra["size"] = shape["half"]
        elif kind == "capsule":
            extra["size"] = [shape["radius"], shape["half_length"], 0]
            mujoco.mju_mulQuat(quat, quat.copy(), np.array(CAPSULE_X_TO_Z))
        else:
            extra["meshname"] = self.add_mesh(spec, shape["mesh"])
        return body.add_geom(type=SHAPES[kind], pos=shape["pos"], quat=quat, mass=0, contype=0, conaffinity=0, **extra)

    def add_body_geoms(self, spec, body, item, contact):
        for shape in item["collision"]:
            geom = self.add_shape(spec, body, shape)
            geom.contype, geom.conaffinity = contact
            geom.friction[0] = shape["friction"]
            geom.group = 3
        for shape in item["visual"]:
            geom = self.add_shape(spec, body, shape)
            geom.group = 2
            geom.material = self.add_material(spec, shape)

    def add_link(self, spec, parent_body, parent, link, children):
        pos, quat = relative_pose(parent, link) if parent is not None else (link["pos"], link["quat"])
        body = parent_body.add_body(name=f"{link['owner']}/{link['name']}", pos=pos, quat=quat)
        if "mass" in link and "joint" in link:
            set_inertia(body, link)
        if "joint" in link:
            joint = link["joint"]
            kind = mujoco.mjtJoint.mjJNT_SLIDE if joint["type"] == "slide" else mujoco.mjtJoint.mjJNT_HINGE
            body.add_joint(name=f"{link['owner']}/{joint['name']}", type=kind, axis=joint["axis"], pos=joint["pos"],
                           range=joint["range"], damping=joint["damping"], frictionloss=joint["friction"])
        self.add_body_geoms(spec, body, link, ARTICULATED)
        for child in children.get(link["name"], []):
            self.add_link(spec, body, link, child, children)

    def add_subset(self, spec, names):
        """Only the actors `names`, without visuals or lights: the fast model used for training."""
        for actor in self.data["actors"]:
            if actor["name"] in names:
                body = spec.worldbody.add_body(name=actor["name"], pos=actor["pos"], quat=actor["quat"])
                if actor["movable"]:
                    body.add_joint(name=actor["name"], type=mujoco.mjtJoint.mjJNT_FREE, damping=PROP_DAMPING)
                    set_inertia(body, actor)
                self.add_body_geoms(spec, body, {**actor, "visual": []}, STATIC)

    def add_to(self, spec):
        world = spec.worldbody
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
        spec.option.sleep_tolerance = SLEEP_TOLERANCE
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.diffuse = [0.05, 0.05, 0.05]
        spec.visual.headlight.specular = [0, 0, 0]
        world.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=[0.3, 0.4, -1.0], castshadow=True,
                        diffuse=[0.42, 0.4, 0.38], specular=[0, 0, 0])
        for x, y in LIGHTS:
            world.add_light(pos=[x, y, 2.4], dir=[0, 0, -1], cutoff=170, castshadow=False, diffuse=[0.1, 0.09, 0.08], specular=[0, 0, 0])
        for actor in self.data["actors"]:
            body = world.add_body(name=actor["name"], pos=actor["pos"], quat=actor["quat"])
            if actor["movable"]:
                body.add_joint(name=actor["name"], type=mujoco.mjtJoint.mjJNT_FREE, damping=PROP_DAMPING)
                set_inertia(body, actor)
            self.add_body_geoms(spec, body, actor, STATIC)
        for articulation in self.data["articulations"]:
            children = {}
            for link in articulation["links"]:
                link["owner"] = articulation["name"]
                children.setdefault(link["parent"], []).append(link)
            for root in children[None]:
                self.add_link(spec, world, None, root, children)


def set_inertia(body, item):
    body.mass = item["mass"]
    body.inertia = item["inertia"]
    body.ipos, body.iquat = item["inertial"]["pos"], item["inertial"]["quat"]
    body.explicitinertial = True
