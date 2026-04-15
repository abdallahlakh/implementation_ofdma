#!/usr/bin/env python3
"""Train code-v2 actor-critic models and save checkpoints."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

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
        with redirect_stdout(io.StringIO()):
            exec(compile(code, filename, "exec"), namespace)

    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save code-v2 checkpoints")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per seed")
    parser.add_argument("--rollout-length", type=int, default=256, help="Rollout length")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="saved_models",
        help="Directory for checkpoints",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = load_code_v2_namespace(root)
    cfg_cls = ns["ActorCriticConfig"]
    agent_cls = ns["ActorCriticAgent"]

    for seed in args.seeds:
        cfg = cfg_cls(
            num_episodes=args.episodes,
            rollout_length=args.rollout_length,
            seed=seed,
        )
        agent = agent_cls(cfg)
        history = agent.train()

        actor_path = out_dir / f"actor_seed{seed}_ep{args.episodes}.pt"
        critic_path = out_dir / f"critic_seed{seed}_ep{args.episodes}.pt"
        meta_path = out_dir / f"meta_seed{seed}_ep{args.episodes}.json"

        torch.save(agent.actor.state_dict(), actor_path)
        torch.save(agent.critic.state_dict(), critic_path)

        metadata = {
            "seed": seed,
            "episodes": args.episodes,
            "rollout_length": args.rollout_length,
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "actor_output_dim": agent.actor_output_dim,
            "final_metrics": {
                "episode_reward": float(history["episode_rewards"][-1]),
                "critic_loss": float(history["critic_loss"][-1]),
                "actor_loss": float(history["actor_loss"][-1]),
                "entropy": float(history["entropy"][-1]),
                "throughput": float(history["throughput"][-1]),
                "collision_rate": float(history["collision_rate"][-1]),
            },
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"saved_actor: {actor_path}")
        print(f"saved_critic: {critic_path}")
        print(f"saved_meta: {meta_path}")


if __name__ == "__main__":
    main()
