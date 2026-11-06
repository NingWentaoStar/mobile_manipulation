"""MuJoCo model of the mobile Panda: chassis read from the URDF, arm and gripper from MuJoCo Menagerie."""
from collections import defaultdict
import math
import re
import xml.etree.ElementTree as E

import mujoco
import numpy as np

from ..geometry import ARM_REST, BASE_CAMERA_SIZE, CAMERA_FOV_Y, GRIPPER_RANGE
from ..paths import MENAGERIE_PANDA, ros_urdf

TIMESTEP = 0.002
CONTROL_SUBSTEPS = 25
TCP_OFFSET = 0.1034
ROOT_ABOVE_FLOOR = {"wheels": 0.205 + 0.002, "planar": 0.0}
GEOM_TYPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
JOINT_TYPES = {"continuous": mujoco.mjtJoint.mjJNT_HINGE, "prismatic": mujoco.mjtJoint.mjJNT_SLIDE}
LOW_FRICTION_LINKS = ("front_support", "rear_support")
BASE_DRIVES = {"wheels": dict(kv=30.0, force=40.0, limit=10.0), "planar": dict(kv=3000.0, force=5000.0, limit=10.0)}
PANDA_NAMES = re.compile(r'((?:name|joint|joint1|joint2|body1|body2)=")(link\d|joint\d|finger_joint\d|hand|left_finger|right_finger)"')
FINGER_BODIES = {"left_finger": "panda_leftfinger", "right_finger": "panda_rightfinger"}
# Camera frames: URDF links look along +x with z up; MuJoCo cameras look along -z with y up.
CAMERA_AXES = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


# Per finger; the actuator drives the mean of both fingers, so it gets twice these gains.
FINGER_STIFFNESS, FINGER_DAMPING = 1000.0, 100.0
FINGER_FRICTION = 1.15
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
REST_POSITIONS = dict(zip(ARM_JOINTS + FINGER_JOINTS, ARM_REST))


def yaw_quat(yaw):
    return [math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)]


def origin(element):
    """(xyz, wxyz) of a URDF element's <origin>."""
    tag = element.find("origin")
    read = lambda key: [float(v) for v in (tag.get(key, "0 0 0") if tag is not None else "0 0 0").split()]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, read("rpy"), "XYZ")
    return np.array(read("xyz")), quat


def compose(first, second):
    position, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(position, quat, first[0], first[1], second[0], second[1])
    return position, quat


def load_panda():
    """Menagerie Panda with the URDF's names; gravity compensation through the joints keeps its weight on the chassis."""
    xml = MENAGERIE_PANDA / "panda.xml"
    if not xml.is_file():
        raise FileNotFoundError(f"{xml} missing; run python scripts/download_menagerie.py")
    text = PANDA_NAMES.sub(lambda m: f'{m[1]}{FINGER_BODIES.get(m[2], "panda_" + m[2])}"', xml.read_text())
    spec = mujoco.MjSpec.from_string(text)
    spec.meshdir = str(MENAGERIE_PANDA / "assets")
    spec.delete(spec.key("home"))
    for body in spec.bodies[1:]:
        body.gravcomp = 1.0
    for joint in spec.joints:
        joint.actgravcomp = True
    for i, actuator in enumerate(spec.actuators[:7], start=1):
        actuator.name = f"panda_joint{i}"
    gripper = spec.actuators[7]
    gripper.name = "gripper"
    gripper.gainprm[0] = FINGER_STIFFNESS * 2
    gripper.biasprm[1:3] = [-FINGER_STIFFNESS * 2, -FINGER_DAMPING * 2]
    gripper.ctrlrange = GRIPPER_RANGE
    spec.body("panda_hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET])
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        for geom in spec.body(finger).geoms:
            geom.friction[0], geom.priority = FINGER_FRICTION, 1
    return spec


def add_link_geoms(body, link):
    name, collides = link.get("name"), link.find("collision") is not None
    for visual in link.findall("visual"):
        shape = visual.find("geometry")[0]
        position, quat = origin(visual)
        if shape.tag == "box":
            size = [float(v)/2 for v in shape.get("size").split()]
        else:
            size = [float(shape.get("radius")), float(shape.get("length", 0))/2, 0.0]
        rgba = [float(v) for v in visual.find("material/color").get("rgba").split()]
        geom = body.add_geom(type=GEOM_TYPES[shape.tag], size=size, pos=position, quat=quat, rgba=rgba, mass=0)
        wheel = collides and name.endswith("_wheel")
        geom.contype = geom.conaffinity = 1 if collides and not wheel else 0
        if collides and name in LOW_FRICTION_LINKS:
            geom.friction[0], geom.priority = 0.015, 1
        if wheel:
            # A cylinder's several ground contacts scrub when turning on the spot.
            radius, half_width = size[:2]
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[radius, half_width, radius], group=3,
                          friction=[1.2, 0.005, 0.0001], priority=1, mass=0, rgba=[0, 0, 0, 0])


def add_link_inertia(body, link):
    inertial = link.find("inertial")
    tensor = inertial.find("inertia")
    body.mass = float(inertial.find("mass").get("value"))
    body.ipos, body.iquat = origin(inertial)
    body.inertia = [float(tensor.get(key)) for key in ("ixx", "iyy", "izz")]
    body.explicitinertial = True


def add_robot(spec, base, x, y, z, yaw):
    root = E.parse(ros_urdf(base)).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = defaultdict(list)
    for joint in root.findall("joint"):
        joints[joint.find("parent").get("link")].append(joint)
    panda = load_panda()

    def add_children(parent, name):
        for joint in joints[name]:
            child = joint.find("child").get("link")
            position, quat = origin(joint)
            if child == "panda_link0":
                frame = parent.add_frame(pos=position, quat=quat)
                spec.meshdir = panda.meshdir
                spec.attach(panda, frame=frame, prefix="", suffix="")
                continue
            if child.startswith("panda_"):
                continue
            body = parent.add_body(name=child, pos=position, quat=quat)
            add_link_inertia(body, links[child])
            add_link_geoms(body, links[child])
            if joint.get("type") in JOINT_TYPES:
                axis = [float(v) for v in joint.find("axis").get("xyz").split()]
                body.add_joint(name=joint.get("name"), type=JOINT_TYPES[joint.get("type")], axis=axis)
                add_base_drive(spec, base, joint.get("name"))
            add_children(body, child)

    root_name = "chassis_link" if base == "wheels" else "world_root"
    body = spec.worldbody.add_body(name=root_name, pos=[x, y, z], quat=yaw_quat(yaw))
    add_link_inertia(body, links[root_name])
    add_link_geoms(body, links[root_name])
    if base == "wheels":
        body.add_freejoint(name="root")
    else:
        body.mocap = True   # plain fixed bodies are cached; reset() must be able to move it
    add_children(body, root_name)
    add_cameras(spec, root)


def add_base_drive(spec, base, joint):
    drive = BASE_DRIVES[base]
    actuator = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    actuator.set_to_velocity(kv=drive["kv"])
    actuator.ctrlrange = [-drive["limit"], drive["limit"]]
    actuator.forcerange = [-drive["force"], drive["force"]]
    actuator.forcelimited = True


def add_cameras(spec, root):
    joint = {j.get("name"): j for j in root.findall("joint")}
    axes = np.zeros(4)
    mujoco.mju_mat2Quat(axes, CAMERA_AXES.reshape(-1))
    frames = {"base_camera": ("panda_link0", [origin(joint["base_camera_joint"])]),
              "hand_camera": ("panda_hand", [origin(joint["hand_camera_mount_joint"]), origin(joint["hand_camera_joint"])])}
    for name, (parent, chain) in frames.items():
        pose = chain[0]
        for step in chain[1:]:
            pose = compose(pose, step)
        position, quat = compose(pose, (np.zeros(3), axes))
        spec.body(parent).add_camera(name=name, pos=position, quat=quat, fovy=math.degrees(CAMERA_FOV_Y))


def add_environment(spec):
    spec.compiler.degree = False
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    spec.visual.global_.offwidth, spec.visual.global_.offheight = BASE_CAMERA_SIZE
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]


def build(add_scene, base, start, extend=None):
    """Compiled (model, spec): add_scene(spec) plus the robot parked at start = (x, y, yaw); extend(spec) adds bodies."""
    spec = mujoco.MjSpec()
    add_environment(spec)
    add_scene(spec)
    if extend is not None:
        extend(spec)
    add_robot(spec, base, start[0], start[1], ROOT_ABOVE_FLOOR[base], start[2])
    return spec.compile(), spec
