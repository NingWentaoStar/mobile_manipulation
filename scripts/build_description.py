"""Generate the mobile Panda URDFs (committed to the repo) from panda_v2.urdf and geometry.py.

    python scripts/build_description.py
"""
from copy import deepcopy
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as E

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobile_panda_sim.geometry import (HEIGHT, RADIUS, TRACK, BASE_CAMERA_EYE, BASE_CAMERA_TARGET,
                                       HAND_MOUNT_XYZ, HAND_MOUNT_RPY, HAND_CAMERA_XYZ)
from mobile_panda_sim.paths import DESCRIPTION, PACKAGE, PANDA_URDF

MESH_URI = f"package://{PACKAGE}/meshes/franka"


# 1 mm above the ground: at 5 mm the chassis rocks and grasps miss, at 0 the wheels slip and it cannot drive.
SUPPORT_RADIUS = 0.039


def link(root, name, shape=None, size=None, mass=1.0, color="0.15 0.22 0.3 1", collision=True):
    out = E.SubElement(root, "link", name=name)
    inert = E.SubElement(out, "inertial")
    E.SubElement(inert, "mass", value=str(mass))
    if shape == "box":
        x, y, z = size
        inertia = [mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12]
    elif shape == "cylinder":
        r, width = size
        inertia = [mass*(3*r*r+width*width)/12, mass*r*r/2, mass*(3*r*r+width*width)/12]
    elif shape == "sphere":
        inertia = [0.4*mass*size**2]*3
    else:
        inertia = [1e-4]*3
    E.SubElement(inert, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]),
                 ixy="0", ixz="0", iyz="0")
    if shape:
        for kind in (["visual", "collision"] if collision else ["visual"]):
            item = E.SubElement(out, kind)
            if shape == "cylinder":
                E.SubElement(item, "origin", xyz="0 0 0", rpy="1.57079632679 0 0")
            geom = E.SubElement(item, "geometry")
            if shape == "box":
                E.SubElement(geom, "box", size=" ".join(map(str, size)))
            elif shape == "cylinder":
                E.SubElement(geom, "cylinder", radius=str(size[0]), length=str(size[1]))
            else:
                E.SubElement(geom, "sphere", radius=str(size))
            if kind == "visual":
                E.SubElement(E.SubElement(item, "material", name=f"{name}_paint"), "color", rgba=color)
    return out


def joint(root, name, parent, child, xyz="0 0 0", rpy="0 0 0", kind="fixed", axis=None):
    j = E.SubElement(root, "joint", name=name, type=kind)
    E.SubElement(j, "parent", link=parent)
    E.SubElement(j, "child", link=child)
    E.SubElement(j, "origin", xyz=xyz, rpy=rpy)
    if axis:
        E.SubElement(j, "axis", xyz=axis)
        if kind == "prismatic":
            E.SubElement(j, "limit", lower="-10", upper="10", effort="5000", velocity="1")
        else:
            E.SubElement(j, "limit", effort="100", velocity="20")


def fmt(values):
    return " ".join(f"{v:.9g}" for v in values)


def look_at_rpy(eye, target):
    """Camera link rotation looking from eye to target (x forward, y left, z up)."""
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(v*v for v in f))
    f = [v/n for v in f]
    up = [0.0, 0.0, 1.0]
    left = [up[1]*f[2]-up[2]*f[1], up[2]*f[0]-up[0]*f[2], up[0]*f[1]-up[1]*f[0]]
    n = math.sqrt(sum(v*v for v in left))
    left = [v/n for v in left]
    upv = [f[1]*left[2]-f[2]*left[1], f[2]*left[0]-f[0]*left[2], f[0]*left[1]-f[1]*left[0]]
    r20, r21, r22 = f[2], left[2], upv[2]
    r10, r00 = f[1], f[0]
    pitch = math.asin(-r20)
    roll = math.atan2(r21, r22)
    yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def camera(root, name, parent, xyz, rpy):
    """<name>_link looks along +x with z up (the simulator mounts the camera here);
    <name>_optical_frame follows ROS REP-103 optical axes (z forward, x right, y down)."""
    link(root, f"{name}_link", mass=0.001)
    joint(root, f"{name}_joint", parent, f"{name}_link", xyz=fmt(xyz), rpy=fmt(rpy))
    link(root, f"{name}_optical_frame", mass=0.001)
    joint(root, f"{name}_optical_joint", f"{name}_link", f"{name}_optical_frame",
          rpy=fmt((-math.pi/2, 0, -math.pi/2)))


def build():
    for mode in ("wheels", "planar"):
        root = E.Element("robot", name=f"mobile_panda_{mode}")
        if mode == "planar":
            for name in ("world_root", "base_x_link", "base_y_link"):
                link(root, name, mass=0.01)
            joint(root, "base_x_joint", "world_root", "base_x_link", kind="prismatic", axis="1 0 0")
            joint(root, "base_y_joint", "base_x_link", "base_y_link", kind="prismatic", axis="0 1 0")
            joint(root, "base_yaw_joint", "base_y_link", "chassis_link", xyz="0 0 0.20", kind="continuous", axis="0 0 1")
        link(root, "chassis_link", "box", [0.55, 0.46, 0.14], mass=40)
        pillar_length = HEIGHT - 0.27
        link(root, "pedestal_link", "box", [0.18, 0.18, pillar_length], mass=4, color="0.7 0.72 0.75 1")
        joint(root, "pedestal_mount", "chassis_link", "pedestal_link", xyz=f"0 0 {0.07+pillar_length/2}")
        joint(root, "arm_mount", "pedestal_link", "panda_link0", xyz=f"0 0 {pillar_length/2}")
        for side, y in (("left", TRACK/2), ("right", -TRACK/2)):
            link(root, f"{side}_wheel", "cylinder", [RADIUS, 0.06], mass=2,
                 color="0.035 0.035 0.04 1", collision=mode == "wheels")
            joint(root, f"{side}_wheel_joint", "chassis_link", f"{side}_wheel", xyz=f"0 {y} -0.105",
                  kind="continuous" if mode == "wheels" else "fixed", axis="0 1 0" if mode == "wheels" else None)
        for side, x in (("front", 0.23), ("rear", -0.23)):
            link(root, f"{side}_support", "sphere", SUPPORT_RADIUS, mass=0.5,
                 color="0.15 0.15 0.15 1", collision=mode == "wheels")
            joint(root, f"{side}_support_mount", "chassis_link", f"{side}_support", xyz=f"{x} 0 -0.165")
        original = E.parse(PANDA_URDF).getroot()
        for child in original:
            item = deepcopy(child)
            for mesh in item.findall(".//mesh"):
                rel = Path(mesh.attrib["filename"]).relative_to("franka_description/meshes")
                mesh.set("filename", f"{MESH_URI}/{rel.as_posix()}")
            root.append(item)
        camera(root, "base_camera", "panda_link0", BASE_CAMERA_EYE, look_at_rpy(BASE_CAMERA_EYE, BASE_CAMERA_TARGET))
        link(root, "hand_camera_mount", mass=0.001)
        joint(root, "hand_camera_mount_joint", "panda_hand", "hand_camera_mount",
              xyz=fmt(HAND_MOUNT_XYZ), rpy=fmt(HAND_MOUNT_RPY))
        camera(root, "hand_camera", "hand_camera_mount", HAND_CAMERA_XYZ, (0, 0, 0))
        E.indent(root)
        tree = E.ElementTree(root)
        out = DESCRIPTION / f"urdf/mobile_panda_{mode}.urdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        print(f"wrote {out}")


if __name__ == "__main__":
    build()
