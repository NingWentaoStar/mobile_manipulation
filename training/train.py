"""PPO for the two skills on the entry console: physics in parallel on the CPU, the network on the GPU if there is one.

    python training/train.py pick --name pick                                  # take an object, hold it at a point
    python training/train.py place --name place --pick-policy skills/policies/pick_move.pt
    python training/train.py evaluate pick skills/policies/pick_move.pt
    python training/train.py evaluate place skills/policies/place.pt --pick-policy skills/policies/pick_move.pt
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobile_panda_sim.skills.learned.tasks import ACTION_SIZE, MAX_STEPS, OBS_SIZE, TARGETS, PickMoveTask, PlaceTask  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer(in_size, out_size, std=np.sqrt(2)):
    linear = nn.Linear(in_size, out_size)
    nn.init.orthogonal_(linear.weight, std)
    nn.init.zeros_(linear.bias)
    return linear


def network(last_size, last_std=np.sqrt(2)):
    return nn.Sequential(layer(OBS_SIZE, 256), nn.Tanh(), layer(256, 256), nn.Tanh(), layer(256, 256), nn.Tanh(), layer(256, last_size, last_std))


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = network(1)
        self.actor_mean = network(ACTION_SIZE, 0.01*np.sqrt(2))
        self.actor_logstd = nn.Parameter(torch.ones(1, ACTION_SIZE) * -0.5)

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor_mean(obs), torch.exp(self.actor_logstd))


def load_agent(path):
    agent = Agent().to(DEVICE)
    agent.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return agent


@torch.no_grad()
def evaluate(actor, make_task, episodes):
    """Share of episodes that succeed, per object; only the first episode of each copy counts."""
    task = make_task(episodes)
    names = list(task.target)
    obs = task.observe()
    won, over = np.zeros(episodes, dtype=bool), np.zeros(episodes, dtype=bool)
    for _ in range(MAX_STEPS):
        action = actor(torch.from_numpy(obs).to(DEVICE)).clamp(-1, 1).cpu().numpy()
        obs, _, failed, ended, _, success = task.step(action)
        won |= success & ~over
        over |= failed | ended
    task.close()
    per_object = {n: round(float(np.mean([w for w, o in zip(won, names) if o == n])), 3) for n in TARGETS if n in names}
    return float(won.mean()), per_object


@torch.no_grad()
def collect_starts(policy, count, seed):
    """States at the end of successful picks: the object is in the hand, held still in the air."""
    actor = load_agent(policy).actor_mean
    task = PickMoveTask(256, seed=seed)
    task.collect, task.saved = True, []
    obs = task.observe()
    while len(task.saved) < count:
        obs = task.step(actor(torch.from_numpy(obs).to(DEVICE)).clamp(-1, 1).cpu().numpy())[0]
    task.close()
    return dict(target=np.array([s[0] for s in task.saved]), qpos=np.array([s[1] for s in task.saved]),
                qvel=np.array([s[2] for s in task.saved]), ctrl=np.array([s[3] for s in task.saved]))


def train(args, make_task):
    torch.manual_seed(args.seed)
    run = Path(args.out) / args.name
    run.mkdir(parents=True, exist_ok=True)
    task = make_task(args.envs, args.seed)
    agent = Agent().to(DEVICE)
    if args.resume:
        agent.load_state_dict(torch.load(args.resume, map_location=DEVICE, weights_only=True))
        agent.actor_logstd.data.clamp_(min=-1.2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    n, length = args.envs, args.rollout
    observations = torch.zeros((length, n, OBS_SIZE), device=DEVICE)
    actions = torch.zeros((length, n, ACTION_SIZE), device=DEVICE)
    log_probs, rewards, values, ends, boots = (torch.zeros((length, n), device=DEVICE) for _ in range(5))
    obs = torch.from_numpy(task.observe()).to(DEVICE)
    steps, start, best, iteration = 0, time.time(), -1.0, 0
    log = open(run / "log.jsonl", "a")
    while steps < args.total_steps:
        iteration += 1
        for t in range(length):
            with torch.no_grad():
                dist = agent.distribution(obs)
                action = dist.sample()
                log_probs[t] = dist.log_prob(action).sum(1)
                values[t] = agent.critic(obs).squeeze(1)
            observations[t], actions[t] = obs, action
            next_obs, reward, failed, ended, final_obs, _ = task.step(action.clamp(-1, 1).cpu().numpy())
            rewards[t] = torch.from_numpy(reward).float().to(DEVICE)
            ends[t] = torch.from_numpy((failed | ended).astype(np.float32)).to(DEVICE)
            boots[t] = 0
            if ended.any():
                # Also after success: valued at zero, hovering by the goal pays more than finishing.
                with torch.no_grad():
                    last = torch.from_numpy(final_obs[ended]).to(DEVICE)
                    boots[t, torch.from_numpy(ended).to(DEVICE)] = agent.critic(last).squeeze(1)
            obs = torch.from_numpy(next_obs).to(DEVICE)
        steps += n * length
        with torch.no_grad():
            advantage, next_value = torch.zeros(n, device=DEVICE), agent.critic(obs).squeeze(1)
            advantages = torch.zeros((length, n), device=DEVICE)
            for t in reversed(range(length)):
                keep = 1 - ends[t]
                delta = rewards[t] + args.gamma * (boots[t] + keep * next_value) - values[t]
                advantage = delta + args.gamma * args.gae_lambda * keep * advantage
                advantages[t], next_value = advantage, values[t]
            returns = advantages + values
        b_obs, b_actions, b_log_probs, b_advantages, b_returns = (
            x.reshape((n * length,) + x.shape[2:]) for x in (observations, actions, log_probs, advantages, returns))
        for _ in range(args.epochs):
            order = torch.randperm(n * length, device=DEVICE)
            stop = False
            for i in range(0, n * length, n * length // args.minibatches):
                index = order[i:i + n * length // args.minibatches]
                dist = agent.distribution(b_obs[index])
                new_log_prob = dist.log_prob(b_actions[index]).sum(1)
                ratio = (new_log_prob - b_log_probs[index]).exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - (new_log_prob - b_log_probs[index])).mean()
                if kl > args.target_kl:
                    stop = True
                    break
                adv = b_advantages[index]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean()
                value_loss = 0.5 * ((agent.critic(b_obs[index]).squeeze(1) - b_returns[index]) ** 2).mean()
                optimizer.zero_grad()
                (policy_loss + 0.5 * value_loss).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
            if stop:
                break
        if iteration % args.eval_every == 0 or steps >= args.total_steps:
            rate, per_object = evaluate(agent.actor_mean, lambda k: make_task(k, args.seed + iteration), args.eval_episodes)
            record = dict(steps=steps, success=round(rate, 4), per_object=per_object, reward=round(float(rewards.mean()), 4),
                          seconds=round(time.time() - start), steps_per_second=round(steps / (time.time() - start)))
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if rate >= best:
                best = rate
                torch.save(agent.state_dict(), run / "best.pt")
    torch.save(agent.state_dict(), run / "final.pt")
    task.close()
    print(f"best success {best:.3f}; checkpoints in {run}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("pick", "place", "evaluate"))
    p.add_argument("positional", nargs="*", help="for evaluate: skill and checkpoint")
    p.add_argument("--name", default="run")
    p.add_argument("--out", default="training/runs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--rollout", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=8_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--target-kl", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--pick-policy", type=Path)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    skill = args.positional[0] if args.command == "evaluate" else args.command
    if skill == "place":
        starts = collect_starts(args.pick_policy, 3000, args.seed)
        make_task = lambda n, seed=args.seed: PlaceTask(n, starts, seed=seed)
    else:
        make_task = lambda n, seed=args.seed: PickMoveTask(n, seed=seed)
    if args.command == "evaluate":
        rate, per_object = evaluate(load_agent(args.positional[1]).actor_mean, make_task, args.episodes)
        print(json.dumps(dict(skill=skill, episodes=args.episodes, success=round(rate, 4), per_object=per_object)))
    else:
        train(args, make_task)


if __name__ == "__main__":
    main()
