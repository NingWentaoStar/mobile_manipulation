"""Classical manipulation skills in the home scene: pick / place, open / close drawers and doors.

Each scenario parks the robot at a fixed base pose (navigation is not part of these skills) and runs a
skill sequence. Every step is checked: skill success, plus any other object moved > 2 cm.

    python scripts/demo_manipulation.py drawer                 # counter drawers
    python scripts/demo_manipulation.py pickplace              # groceries on the entry console
    python scripts/demo_manipulation.py navigate               # drive to the console, then pick and place
    python scripts/demo_manipulation.py fridge                 # fridge door, 35 deg
    python scripts/demo_manipulation.py all --repeat 3 --report outputs/manipulation.json
    python scripts/demo_manipulation.py fridge --record outputs/fridge.mp4
    python scripts/demo_manipulation.py drawer --viewer        # watch in the MuJoCo window (close it to stop)
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills.mj.sim import MujocoScene  # noqa: E402
from skills.library import SkillLibrary  # noqa: E402

SCENARIOS = {
    "drawer": [
        ((-1.15, 1.83, 1.989), [("open", "kitchen_counter", "middle_slide_top"), ("close", "kitchen_counter", "middle_slide_top"), ("reset_arm",)]),
        ((-2.899, 1.86, 1.153), [("open", "kitchen_counter", "left_slide_top"), ("close", "kitchen_counter", "left_slide_top"), ("reset_arm",)]),
    ],
    "pickplace": [
        ((-2.95, -1.15, 3.1416), [("pick", "sugar_box"), ("place", -3.40, -1.25), ("reset_arm",),
                                  ("pick", "pear"), ("place", -3.40, -1.00), ("reset_arm",),
                                  ("pick", "lemon"), ("place", -3.38, -1.42), ("reset_arm",)]),
    ],
    "fridge": [
        ((0.249, 1.492, 1.107), [("open", "fridge", "top_door_hinge"), ("close", "fridge", "top_door_hinge"), ("reset_arm",)]),
    ],
    "navigate": [
        ((-2.55, -1.95, 1.5708), [("navigate", -2.95, -1.15, 3.1416), ("pick", "lemon")]),
    ],
}
SETTLING = {"frying_pan", "apple", "orange", "fruit_bowl"}


class WindowClosed(Exception):
    pass


class Run:
    def __init__(self, args):
        self.args = args
        self.scene = MujocoScene(args.base, viewer=args.viewer)
        self.frames, self.recording = [], False
        self.skills = SkillLibrary(self.scene, on_step=self.on_step, on_idle=self.show if self.scene.viewer else None)

    def on_step(self):
        self.show()
        if self.recording and self.skills.context.steps % 2 == 0:
            self.frames.append(self.scene.capture_rgb()["base_camera"])

    def show(self):
        if self.scene.viewer is not None and not self.scene.viewer.is_running():
            raise WindowClosed

    def close(self):
        self.scene.close()


def call(skills, step):
    name, *params = step
    if name == "pick":
        return skills.run("pick", object=params[0])
    if name == "place":
        return skills.run("place", target=params, frame="map")
    if name in ("open", "close"):
        return skills.run(name, articulation=params[0], joint=params[1])
    if name == "navigate":
        return skills.run("navigate", target=params)
    return skills.run(name)


def run(sim, start, steps, record=None):
    """One scenario: park the robot at `start` (reset), run the skill steps."""
    skills = sim.skills
    skills.reset(start)
    sim.frames, sim.recording = [], record is not None
    world = skills.context.world
    results, target = [], None
    for step in steps:
        name, *params = step
        before = world.object_positions()
        target = params[0] if name == "pick" else target
        t = time.time()
        result = call(skills, step)
        after = world.object_positions()
        moved = {n: round(float(np.linalg.norm(after[n] - before[n])), 3) for n in after
                 if n not in SETTLING and n != target and np.linalg.norm(after[n] - before[n]) > 0.02}
        # Navigation has no obstacle avoidance, so nudging props while driving is not a failure.
        judged = name != "navigate"
        record_ = dict(step=":".join(map(str, step)), success=result.success, message=result.message,
                       disturbed=moved, judged=judged, seconds=round(time.time() - t, 1), info=result.info)
        results.append(record_)
        print(json.dumps({k: v for k, v in record_.items() if k != "info"}), flush=True)
    if record is not None and sim.frames:
        import imageio
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(record), sim.frames, fps=20)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", choices=[*SCENARIOS, "all"])
    p.add_argument("--base", choices=("wheels", "planar"), default="wheels")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--record", type=Path, help="video of the first run of the first scenario")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    report, first = [], True
    sim = Run(args)
    try:
        for name in names:
            for start, steps in SCENARIOS[name]:
                for i in range(args.repeat):
                    results = run(sim, start, steps, record=args.record if first else None)
                    first = False
                    ok = all(r["success"] and (not r["disturbed"] or not r["judged"]) for r in results)
                    report.append(dict(scenario=name, start=start, run=i, ok=ok, steps=results))
        if args.viewer:
            print("done; close the viewer window to exit", flush=True)
            while True:
                sim.show()
                time.sleep(1 / 60)
    except (WindowClosed, KeyboardInterrupt):
        print("viewer closed; stopping", flush=True)
    finally:
        sim.close()
    summary = {}
    for r in report:
        for s in r["steps"]:
            key = f"{r['scenario']}/{s['step']}"
            done, total = summary.get(key, (0, 0))
            summary[key] = (done + (s["success"] and (not s["disturbed"] or not s["judged"])), total + 1)
    print(json.dumps({k: f"{a}/{b}" for k, (a, b) in summary.items()}, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(summary={k: f"{a}/{b}" for k, (a, b) in summary.items()},
                                               runs=report), indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
