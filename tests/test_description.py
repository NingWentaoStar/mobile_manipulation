"""The committed URDFs and their meshes agree."""
import xml.etree.ElementTree as E

import pytest

from mobile_panda_sim.paths import DESCRIPTION, PACKAGE, ros_urdf

ACTIVE = {"wheels": 11, "planar": 12}
CAMERA_FRAMES = {"base_camera_link", "base_camera_optical_frame", "hand_camera_link", "hand_camera_optical_frame"}


@pytest.mark.parametrize("mode", ["wheels", "planar"])
def test_urdf_meshes_and_joints(mode):
    root = E.parse(ros_urdf(mode)).getroot()
    for mesh in root.findall(".//mesh"):
        uri = mesh.attrib["filename"]
        assert uri.startswith(f"package://{PACKAGE}/"), uri
        assert (DESCRIPTION / uri[len(f"package://{PACKAGE}/"):]).is_file(), uri
    active = [j.attrib["name"] for j in root.findall("joint") if j.attrib["type"] != "fixed"]
    assert len(active) == ACTIVE[mode]
    assert CAMERA_FRAMES <= {l.attrib["name"] for l in root.findall("link")}
    assert all(float(m.attrib["value"]) > 0 for m in root.findall(".//mass"))
