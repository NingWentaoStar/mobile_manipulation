"""Download the Franka Panda model from MuJoCo Menagerie (Apache-2.0) into assets/, at a pinned commit."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills.paths import MENAGERIE_PANDA  # noqa: E402

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
FOLDER = "franka_emika_panda"


def main():
    if (MENAGERIE_PANDA / "panda.xml").is_file():
        print(f"already present: {MENAGERIE_PANDA}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *args: subprocess.run(["git", "-C", tmp, *args], check=True)
        git("init", "-q")
        git("remote", "add", "origin", REPO)
        git("sparse-checkout", "set", FOLDER)
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
        MENAGERIE_PANDA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / FOLDER, MENAGERIE_PANDA)
    print(f"saved {MENAGERIE_PANDA}")


if __name__ == "__main__":
    main()
