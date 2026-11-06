"""Two tasks on the entry console of the home scene, with many copies stepped in parallel on the CPU.

PickMoveTask: take an object from the console and hold it still at a point in the air.
PlaceTask: starting with the object in the hand, set it down at a point on the console, let go and lift the hand.
Moving any other object more than 2 cm fails the episode.
"""
from concurrent.futures import ThreadPoolExecutor
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ...geometry import ARM_REST, GRIPPER_RANGE, MAX_JOINT_DELTA  # noqa: E402
from ...mj.kitchen import KitchenScene  # noqa: E402
from ...mj.model import ARM_JOINTS, CONTROL_SUBSTEPS, FINGER_JOINTS, build  # noqa: E402

OBS_SIZE, ACTION_SIZE = 45, 8
MAX_STEPS = 60
TARGETS = ("lemon", "pear", "soup_can")
NEIGHBOURS = ("cracker_box", "sugar_box", "coffee_can", "grocery_basket")
ACTORS = ("floor", "entry_console", *TARGETS, *NEIGHBOURS)
TABLE_Z = 0.918
DOCK = (-2.78, -1.43, np.pi)
ZONE_X, ZONE_Y = (-3.46, -3.30), (-1.62, -1.26)
LIFT_RANGE = (0.02, 0.25)
CLEARANCE = 0.10
AT_GOAL, STILL, DISTURBED = 0.025, 0.2, 0.02
MIN_FINGER_GAP = 0.03
OPEN_GAP = 0.06
HAND_ABOVE = 0.07


def is_grasped(model, data, target_geoms, finger_geoms, finger_q):
    """Both fingers press on the object and it is between them; closed fingertips on top must not count."""
    n = data.ncon
    if n == 0 or data.qpos[finger_q].sum() < MIN_FINGER_GAP:
        return False
    first, second, dist = data.contact.geom1[:n], data.contact.geom2[:n], data.contact.dist[:n]
    pressed = dist < 0
    seen = [False, False]
    for a, b in zip(first[pressed], second[pressed]):
        if a in target_geoms or b in target_geoms:
            other = b if a in target_geoms else a
            for i, geoms in enumerate(finger_geoms):
                if other in geoms:
                    seen[i] = True
    return seen[0] and seen[1]


def body_geoms(model, body):
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g])}


def body_box(model, data, body):
    """(centre in the body frame, half extents) of a body's collision meshes, aligned with the world at rest."""
    points = []
    for geom in body_geoms(model, body):
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points.append(data.geom_xpos[geom] + model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T)
    points = np.concatenate(points)
    low, high = points.min(0), points.max(0)
    return data.xmat[body].reshape(3, 3).T @ ((low + high)/2 - data.xpos[body]), (high - low)/2


class ConsoleTask:
    def __init__(self, num_envs, seed=0, threads=None):
        from ...mj.sim import MujocoScene
        scene = MujocoScene("planar", DOCK)
        self.rest_poses = {name: (position.copy(), quat.copy()) for name, (position, quat) in scene.settled.items()}
        scene.close()
        kitchen = KitchenScene.load()
        self.model, _ = build(lambda spec: kitchen.add_subset(spec, ACTORS), "planar", DOCK)
        model = self.model
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.datas = [mujoco.MjData(model) for _ in range(num_envs)]
        threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self.pool = ThreadPoolExecutor(threads)
        self.chunks = [self.datas[i::threads] for i in range(threads) if self.datas[i::threads]]
        self.arm_q = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.arm_v = [model.joint(n).dofadr[0] for n in ARM_JOINTS]
        self.finger_q = [model.joint(n).qposadr[0] for n in FINGER_JOINTS]
        self.finger_v = [model.joint(n).dofadr[0] for n in FINGER_JOINTS]
        self.arm_ctrl = [model.actuator(n).id for n in ARM_JOINTS]
        self.gripper = model.actuator("gripper").id
        self.tcp = model.site("tcp").id
        self.finger_geoms = [body_geoms(model, model.body(n).id) for n in ("panda_leftfinger", "panda_rightfinger")]
        self.bodies = {name: model.body(name).id for name in (*TARGETS, *NEIGHBOURS)}
        self.addresses = {name: model.joint(name).qposadr[0] for name in self.bodies}
        self.target_geoms = {name: body_geoms(model, self.bodies[name]) for name in TARGETS}
        data = self.datas[0]
        self.place_all(data)
        mujoco.mj_forward(model, data)
        boxes = {name: body_box(model, data, self.bodies[name]) for name in TARGETS}
        self.centres = {name: box[0] for name, box in boxes.items()}
        self.sizes = {name: box[1] for name, box in boxes.items()}
        self.target = [TARGETS[0]] * num_envs
        self.goal = np.zeros((num_envs, 3))
        self.home = np.zeros((num_envs, len(self.bodies), 3))
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.grasped = np.zeros(num_envs, dtype=bool)
        for i in range(num_envs):
            self.reset_one(i)

    def place_all(self, data):
        for name, address in self.addresses.items():
            position, quat = self.rest_poses[name]
            data.qpos[address:address + 3], data.qpos[address + 3:address + 7] = position, quat

    def free_spot(self, skip):
        """A point of the working zone at least CLEARANCE away from every object but `skip`."""
        others = [self.rest_poses[n][0][:2] for n in self.bodies if n != skip]
        while True:
            spot = np.array([self.rng.uniform(*ZONE_X), self.rng.uniform(*ZONE_Y)])
            if all(np.linalg.norm(spot - other) > CLEARANCE for other in others):
                return spot

    def spawn(self, i):
        """Rest pose of everything, the target moved to a free spot of the zone with a random yaw."""
        model, data, rng = self.model, self.datas[i], self.rng
        mujoco.mj_resetData(model, data)
        name = TARGETS[rng.integers(len(TARGETS))]
        self.target[i] = name
        self.place_all(data)
        data.qpos[self.arm_q + self.finger_q] = ARM_REST
        data.qpos[self.arm_q] += rng.normal(0, 0.02, 7)
        address = self.addresses[name]
        data.qpos[address:address + 2] = self.free_spot(name)
        yaw = rng.uniform(0, 2*np.pi)
        turned = np.zeros(4)
        mujoco.mju_mulQuat(turned, np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]), self.rest_poses[name][1])
        data.qpos[address + 3:address + 7] = turned
        data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q]
        data.ctrl[self.gripper] = GRIPPER_RANGE[1]
        data.mocap_pos[0], data.mocap_quat[0] = [DOCK[0], DOCK[1], 0.0], [np.cos(DOCK[2]/2), 0, 0, np.sin(DOCK[2]/2)]

    def finish_reset(self, i):
        data = self.datas[i]
        mujoco.mj_forward(self.model, data)
        self.home[i] = [data.xpos[self.bodies[n]] for n in self.bodies]
        self.steps[i] = 0
        self.grasped[i] = False

    def _advance(self, datas):
        for data in datas:
            mujoco.mj_step(self.model, data, CONTROL_SUBSTEPS)

    def centre(self, i):
        data, name = self.datas[i], self.target[i]
        body = self.bodies[name]
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ self.centres[name]

    def observe(self):
        obs = np.zeros((self.num_envs, OBS_SIZE), dtype=np.float32)
        tcp_quat = np.zeros(4)
        for i, data in enumerate(self.datas):
            name = self.target[i]
            self.grasped[i] = is_grasped(self.model, data, self.target_geoms[name], self.finger_geoms, self.finger_q)
            centre, quat, tcp = self.centre(i), data.xquat[self.bodies[name]], data.site_xpos[self.tcp]
            mujoco.mju_mat2Quat(tcp_quat, data.site_xmat[self.tcp])
            obs[i, :7], obs[i, 7:9] = data.qpos[self.arm_q], data.qpos[self.finger_q]
            obs[i, 9:16], obs[i, 16:18] = data.qvel[self.arm_v], data.qvel[self.finger_v]
            obs[i, 18] = self.grasped[i]
            obs[i, 19:22], obs[i, 22:26] = tcp, tcp_quat
            obs[i, 26:29] = self.goal[i]
            obs[i, 29:32], obs[i, 32:36] = centre, quat
            obs[i, 36:39], obs[i, 39:42] = centre - tcp, self.goal[i] - centre
            obs[i, 42:45] = self.sizes[name]
        return obs

    def disturbed(self):
        moved = np.zeros(self.num_envs, dtype=bool)
        names = list(self.bodies)
        for i, data in enumerate(self.datas):
            for k, name in enumerate(names):
                if name != self.target[i] and np.linalg.norm(data.xpos[self.bodies[name]] - self.home[i, k]) > DISTURBED:
                    moved[i] = True
                    break
        return moved

    def reset(self):
        for i in range(self.num_envs):
            self.reset_one(i)
        return self.observe()

    def step(self, actions):
        """Returns obs, reward, failed (end with no future value), ended (end by success or time), final_obs, success."""
        low, high = GRIPPER_RANGE
        for i, data in enumerate(self.datas):
            data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q] + actions[i, :7]*MAX_JOINT_DELTA
            data.ctrl[self.gripper] = low + (actions[i, 7] + 1)/2*(high - low)
        list(self.pool.map(self._advance, self.chunks))
        self.steps += 1
        obs = self.observe()
        reward, success, failed = self.score(obs)
        moved = self.disturbed()
        failed = (failed | moved) & ~success
        reward = np.where(failed, reward - 0.5, reward)
        ended = success | ((self.steps >= MAX_STEPS) & ~failed)
        final_obs = obs.copy()
        done = failed | ended
        if done.any():
            self.on_end(success)
            for i in np.flatnonzero(done):
                self.reset_one(i)
            obs = self.observe()
        return obs, reward, failed, ended, final_obs, success

    def on_end(self, success):
        pass

    def close(self):
        self.pool.shutdown()


class PickMoveTask(ConsoleTask):
    """collect=True keeps the state of every success, the starting points of PlaceTask."""
    collect = False

    def reset_one(self, i):
        self.spawn(i)
        low, high = LIFT_RANGE
        self.goal[i] = [self.rng.uniform(*ZONE_X), self.rng.uniform(*ZONE_Y), TABLE_Z + self.rng.uniform(low, high)]
        self.finish_reset(i)

    def score(self, obs):
        reach = np.linalg.norm(obs[:, 36:39], axis=1)
        to_goal = np.linalg.norm(obs[:, 39:42], axis=1)
        speed = np.linalg.norm(obs[:, 9:16], axis=1)
        arrived = to_goal <= AT_GOAL
        success = arrived & (speed < STILL) & self.grasped
        # The sharp term makes the hand go down to the object's centre: round objects squeeze out of a grip above it.
        carry = 3*(1 - np.tanh(5*to_goal)) + 2*(1 - np.tanh(30*to_goal))
        reward = 1 - np.tanh(5*reach) + 1 - np.tanh(30*reach) + self.grasped + carry*self.grasped + 2*(1 - np.tanh(5*speed))*arrived
        reward[success] = 10.0
        return reward/10.0, success, np.zeros(self.num_envs, dtype=bool)

    def on_end(self, success):
        if self.collect:
            for i in np.flatnonzero(success):
                data = self.datas[i]
                self.saved.append((TARGETS.index(self.target[i]), data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))


class PlaceTask(ConsoleTask):
    def __init__(self, num_envs, starts, seed=0, threads=None):
        self.starts = starts
        super().__init__(num_envs, seed, threads)

    def reset_one(self, i):
        model, data, rng = self.model, self.datas[i], self.rng
        target, qpos, qvel, ctrl = (self.starts[key] for key in ("target", "qpos", "qvel", "ctrl"))
        pick = rng.integers(len(target))
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos[pick], qvel[pick], ctrl[pick]
        data.mocap_pos[0], data.mocap_quat[0] = [DOCK[0], DOCK[1], 0.0], [np.cos(DOCK[2]/2), 0, 0, np.sin(DOCK[2]/2)]
        name = TARGETS[target[pick]]
        self.target[i] = name
        self.goal[i] = [*self.free_spot(name), TABLE_Z + self.sizes[name][2]]
        self.finish_reset(i)

    def score(self, obs):
        to_goal = np.linalg.norm(obs[:, 39:42], axis=1)
        gap = obs[:, 7:9].sum(1)
        above = -obs[:, 38]
        object_speed = np.array([np.linalg.norm(data.cvel[self.bodies[self.target[i]]]) for i, data in enumerate(self.datas)])
        near = to_goal <= AT_GOAL
        released = near & ~self.grasped
        success = released & (gap > OPEN_GAP) & (above > HAND_ABOVE) & (object_speed < 0.1)
        reward = (3*(1 - np.tanh(5*to_goal)) + 2*(1 - np.tanh(30*to_goal)) + 2*near*np.clip(gap/0.08, 0, 1) +
                  3*released*(1 - np.tanh(10*np.maximum(HAND_ABOVE - above, 0))))
        reward[success] = 10.0
        dropped = ~self.grasped & (to_goal > 2*CLEARANCE/3) & (self.steps > 2)
        return reward/10.0, success, dropped
