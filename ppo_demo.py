"""
Minimal PPO demo on a toy 1-D navigation task.

The script uses a small custom environment so it can run without extra
dependencies. It is intentionally compact and readable for educational
purposes, not performance.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class LineWorldEnv:
    """A tiny 1-D navigation environment.

    The agent starts at position 0 and must reach +5. It receives -0.01
    per step to encourage shorter paths and +1.0 for reaching the goal.
    """

    def __init__(self, goal: int = 5, max_steps: int = 40) -> None:
        self.goal = goal
        self.max_steps = max_steps
        self.position = 0
        self.steps = 0

    def reset(self) -> torch.Tensor:
        self.position = 0
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        # Observations are normalized to [-1, 1] to help learning.
        return torch.tensor([self.position / float(self.goal)], dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        self.position += -1 if action == 0 else 1
        self.steps += 1

        done = False
        reward = -0.01
        if self.position >= self.goal:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.shared(state)
        logits = self.policy_head(base)
        value = self.value_head(base).squeeze(-1)
        return logits, value


@dataclass
class Trajectory:
    states: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]


def collect_trajectory(env: LineWorldEnv, model: PolicyValueNet, device: torch.device) -> Trajectory:
    state = env.reset().to(device)
    states: List[torch.Tensor] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    done = False
    while not done:
        logits, value = model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        states.append(state)
        actions.append(int(action.item()))
        log_probs.append(dist.log_prob(action))
        values.append(value)

        next_state, reward, done = env.step(actions[-1])
        state = next_state.to(device)
        rewards.append(reward)
        dones.append(done)

    return Trajectory(states, actions, rewards, dones, log_probs, values)


def compute_gae(
    rewards: List[float],
    values: List[torch.Tensor],
    dones: List[bool],
    gamma: float,
    lam: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = []
    gae = 0.0
    values = values + [torch.tensor(0.0, device=device)]

    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)

    advantages_tensor = torch.stack(advantages)
    returns = advantages_tensor + torch.stack(values[:-1])
    return advantages_tensor.detach(), returns.detach()


def ppo_update(
    trajectories: List[Trajectory],
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    clip_eps: float,
    epochs: int,
    batch_size: int,
    gamma: float,
    lam: float,
    device: torch.device,
) -> None:
    # Flatten trajectories
    states = torch.cat([torch.stack(traj.states) for traj in trajectories]).to(device)
    actions = torch.tensor([a for traj in trajectories for a in traj.actions], device=device)
    old_log_probs = torch.stack([lp for traj in trajectories for lp in traj.log_probs]).to(device)
    values = [v for traj in trajectories for v in traj.values]
    rewards = [r for traj in trajectories for r in traj.rewards]
    dones = [d for traj in trajectories for d in traj.dones]

    advantages, returns = compute_gae(rewards, values, dones, gamma, lam, device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = states.size(0)
    for _ in range(epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            logits, value_pred = model(states[batch_idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[batch_idx])

            ratio = (new_log_probs - old_log_probs[batch_idx]).exp()
            surr1 = ratio * advantages[batch_idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns[batch_idx] - value_pred).pow(2).mean()
            entropy_bonus = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def train(
    episodes: int,
    batch_size: int,
    lr: float,
    clip_eps: float,
    update_epochs: int,
    gamma: float,
    lam: float,
    seed: int,
    device: torch.device,
) -> PolicyValueNet:
    set_seed(seed)
    env = LineWorldEnv()
    model = PolicyValueNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trajectories: List[Trajectory] = []
    recent_returns: List[float] = []
    for episode in range(1, episodes + 1):
        traj = collect_trajectory(env, model, device)
        trajectories.append(traj)
        recent_returns.append(sum(traj.rewards))
        recent_returns = recent_returns[-20:]

        if len(trajectories) >= batch_size:
            ppo_update(
                trajectories,
                model,
                optimizer,
                clip_eps=clip_eps,
                epochs=update_epochs,
                batch_size=len(torch.cat([torch.stack(t.states) for t in trajectories])),
                gamma=gamma,
                lam=lam,
                device=device,
            )
            trajectories = []

        if episode % 20 == 0:
            mean_return = sum(recent_returns) / len(recent_returns)
            print(f"Episode {episode:04d} | Recent mean return: {mean_return:.3f}")

    return model


def evaluate(env: LineWorldEnv, model: PolicyValueNet, device: torch.device) -> float:
    state = env.reset().to(device)
    total_reward = 0.0
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = model(state)
            action = torch.argmax(logits).item()
        state, reward, done = env.step(action)
        state = state.to(device)
        total_reward += reward
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PPO demo on a 1-D task")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to collect")
    parser.add_argument("--batch-size", type=int, default=8, help="Trajectories per PPO update")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Optimizer learning rate")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--update-epochs", type=int, default=4, help="Gradient steps per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device to use (cpu or cuda)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = train(
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        clip_eps=args.clip_eps,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        lam=args.gae_lambda,
        seed=args.seed,
        device=device,
    )

    env = LineWorldEnv()
    reward = evaluate(env, model, device)
    print(f"Evaluation return after training: {reward:.3f}")


if __name__ == "__main__":
    main()
