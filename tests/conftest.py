import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobile_panda_sim.paths import MENAGERIE_PANDA  # noqa: E402

collect_ignore = []
if importlib.util.find_spec("mujoco") is None or not (MENAGERIE_PANDA / "panda.xml").is_file():
    collect_ignore.append("test_mujoco.py")
