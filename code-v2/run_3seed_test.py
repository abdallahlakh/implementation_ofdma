#!/usr/bin/env python3
"""Run a full 3-seed actor-critic test using only code-v2 files.

This script:
1) Loads split code-v2 modules into one shared namespace
2) Trains the agent for each seed
3) Aggregates mean/std metrics
4) Saves a 2x2 summary plot
5) Prints key summary numbers
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.distributions import Categorical
from typing import NamedTuple


def load_code_v2_namespace(root: Path) -> dict:
    """Execute split code-v2 files into one namespace."""
    namespace = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "optim": optim,
        "Categorical": Categorical,
        "NamedTuple": NamedTuple,
        "dataclass": dataclass,
    }

    ordered_files = [
        "ofdma_channel.py",
        "odfma_environment_definition.py",
        "actor_critic_definition.py",
        "trajectory_definition.py",
        "actor_critic_algorithm.py",
    ]

    for filename in ordered_files:
        code = (root / filename).read_text(encoding="utf-8")
        # Suppress example prints from module-level demo code.
        with redirect_stdout(io.StringIO()):
            exec(compile(code, filename, "exec"), namespace)

    return namespace


def run_single_seed(agent_cls, cfg_cls, episodes: int, rollout_length: int, seed: int) -> dict:
    cfg = cfg_cls(num_episodes=episodes, rollout_length=rollout_length, seed=seed)
    agent = agent_cls(cfg)
    return agent.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-seed code-v2 training test")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per seed")
    parser.add_argument("--rollout-length", type=int, default=256, help="Rollout length")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="List of random seeds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_results_code_v2_stable_3seeds.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ns = load_code_v2_namespace(root)

    cfg_cls = ns["ActorCriticConfig"]
    agent_cls = ns["ActorCriticAgent"]

    all_rewards = []
    all_critic = []
    all_actor = []
    all_entropy = []
    all_throughput = []
    all_collision_rate = []

    for seed in args.seeds:
        history = run_single_seed(
            agent_cls=agent_cls,
            cfg_cls=cfg_cls,
            episodes=args.episodes,
            rollout_length=args.rollout_length,
            seed=seed,
        )
        all_rewards.append(np.array(history["episode_rewards"], dtype=np.float32))
        all_critic.append(np.array(history["critic_loss"], dtype=np.float32))
        all_actor.append(np.array(history["actor_loss"], dtype=np.float32))
        all_entropy.append(np.array(history["entropy"], dtype=np.float32))
        all_throughput.append(np.array(history["throughput"], dtype=np.float32))
        all_collision_rate.append(np.array(history["collision_rate"], dtype=np.float32))

    rewards = np.stack(all_rewards)
    critic = np.stack(all_critic)
    actor = np.stack(all_actor)
    entropy = np.stack(all_entropy)
    throughput = np.stack(all_throughput)
    collision_rate = np.stack(all_collision_rate)

    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)
    mean_critic = critic.mean(axis=0)
    mean_actor = actor.mean(axis=0)
    mean_entropy = entropy.mean(axis=0)
    mean_throughput = throughput.mean(axis=0)
    mean_collision_rate = collision_rate.mean(axis=0)

    x = np.arange(1, rewards.shape[1] + 1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    axes[0, 0].plot(x, mean_rewards, label="Mean Reward")
    axes[0, 0].fill_between(
        x,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="+-1 std",
    )
    axes[0, 0].set_title("Episode Reward (Seed Mean +- Std)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(x, mean_critic)
    axes[0, 1].set_title("Critic Loss (Mean)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    axes[1, 0].plot(x, mean_actor)
    axes[1, 0].set_title("Actor Loss (Mean)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)

    axes[1, 1].plot(x, mean_entropy)
    axes[1, 1].set_title("Policy Entropy (Mean)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].grid(True)

    axes[2, 0].plot(x, mean_throughput)
    axes[2, 0].set_title("Throughput (Mean)")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Successful Allocations / Step")
    axes[2, 0].grid(True)

    axes[2, 1].plot(x, mean_collision_rate)
    axes[2, 1].set_title("Collision Rate (Mean)")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("Failed Attempts / Attempts")
    axes[2, 1].set_ylim(0.0, 1.0)
    axes[2, 1].grid(True)

    plt.tight_layout()
    out = root / args.output
    plt.savefig(out, dpi=150)

    print("saved_plot:", out)
    print("seeds:", args.seeds)
    print("episodes_per_seed:", args.episodes)
    print("rollout_length:", args.rollout_length)
    print("reward_first10_mean:", float(mean_rewards[:10].mean()))
    print("reward_last10_mean:", float(mean_rewards[-10:].mean()))
    print("reward_last10_std_across_seeds:", float(rewards[:, -10:].mean(axis=1).std()))
    print("critic_final_mean:", float(mean_critic[-1]))
    print("entropy_final_mean:", float(mean_entropy[-1]))
    print("throughput_final_mean:", float(mean_throughput[-1]))
    print("collision_rate_final_mean:", float(mean_collision_rate[-1]))


if __name__ == "__main__":
    main()
