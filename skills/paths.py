from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "mobile_panda_description"
DESCRIPTION = REPO / "ros2_sim_ws/src" / PACKAGE
POLICIES = REPO / "skills/policies"
OUTPUTS = REPO / "outputs"
PANDA_URDF = REPO / "skills/manipulation/panda/panda_v2.urdf"
MENAGERIE_PANDA = REPO / "assets/mujoco_menagerie/franka_emika_panda"
KITCHEN = REPO / "assets/mujoco_kitchen"


def ros_urdf(mode):
    return DESCRIPTION / f"urdf/mobile_panda_{mode}.urdf"
