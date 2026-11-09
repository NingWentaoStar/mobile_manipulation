"""The trained skills in the home scene: drive to the entry console, take an object, hold it at a point, set it down.

    python scripts/demo_rl_skills.py --object pear                      # with the viewer window
    python scripts/demo_rl_skills.py --no-viewer --episodes 20 --report outputs/rl_skills.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills.geometry import ARM_REST  # noqa: E402
from skills.mj.sim import MujocoScene  # noqa: E402
from skills.library import SkillLibrary  # noqa: E402
from skills.library.learned.tasks import CLEARANCE, DOCK, NEIGHBOURS, TABLE_Z, TARGETS, ZONE_X, ZONE_Y  # noqa: E402

HOLD_HEIGHT = 0.15


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def base_pose(scene):
    position, q = scene.pose(scene.chassis)
    return np.array([position[0], position[1], math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))])


def hold_arm(scene):
    q = scene.joint_positions()[-9:-2]
    return np.clip(np.array(ARM_REST[:7]) - q, -0.07, 0.07)


def drive_to(scene, target, yaw, on_step, position_tolerance=0.025, yaw_tolerance=0.035):
    """Point-to-point driving with no obstacle avoidance: the way to the console is clear."""
    for _ in range(800):
        p = base_pose(scene)
        delta = np.asarray(target) - p[:2]
        distance = float(np.linalg.norm(delta))
        if distance < position_tolerance:
            error = wrap(yaw - p[2])
            if abs(error) < yaw_tolerance:
                for _ in range(20):
                    scene.step(0, 0, hold_arm(scene), 0.04)
                return distance, error
            velocity = (0.0, math.copysign(min(max(abs(error)*1.8, 0.12), 0.5), error))
        else:
            heading = wrap(math.atan2(delta[1], delta[0]) - p[2])
            velocity = (min(0.22, max(distance*0.9, 0.035)) if abs(heading) < 0.3 else 0.0, float(np.clip(heading*2.0, -0.6, 0.6)))
        scene.step(velocity[0], velocity[1], hold_arm(scene), 0.04)
        on_step()
    raise RuntimeError("did not reach the docking pose")


def free_spot(rng, scene, skip):
    others = [scene.settled[n][0][:2] for n in (*TARGETS, *NEIGHBOURS) if n != skip]
    while True:
        spot = np.array([rng.uniform(*ZONE_X), rng.uniform(*ZONE_Y)])
        if all(np.linalg.norm(spot - other) > CLEARANCE for other in others):
            return spot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", choices=TARGETS, help="default: each in turn")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", choices=("wheels", "planar"), default="wheels")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    scene = MujocoScene(args.base, viewer=not args.no_viewer)
    on_step = (lambda: time.sleep(scene.control_timestep)) if not args.no_viewer else (lambda: None)
    skills = SkillLibrary(scene, on_step=on_step)
    rng = np.random.default_rng(args.seed)
    results = []
    for episode in range(args.episodes):
        name = args.object or TARGETS[episode % len(TARGETS)]
        skills.reset()
        address = scene.model.joint(name).qposadr[0]
        scene.data.qpos[address:address + 2] = free_spot(rng, scene, name)
        drive_to(scene, DOCK[:2], DOCK[2], on_step)
        spot = free_spot(rng, scene, name)
        goal = [spot[0], spot[1], TABLE_Z + HOLD_HEIGHT]
        pick = skills.run("rl_pick_move", object=name, goal=goal)
        place = skills.run("rl_place", target=free_spot(rng, scene, name)) if pick else None
        record = dict(episode=episode, object=name, base=base_pose(scene).round(3).tolist(), pick_move=vars(pick), place=vars(place) if place else None)
        results.append(record)
        print(json.dumps(record), flush=True)
    picks = sum(r["pick_move"]["success"] for r in results)
    places = sum(bool(r["place"] and r["place"]["success"]) for r in results)
    print(json.dumps(dict(episodes=len(results), pick_move=picks, place=places)), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if scene.viewer is not None:
        while scene.viewer.is_running():
            time.sleep(0.1)
    scene.close()
    return 0 if picks == places == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
