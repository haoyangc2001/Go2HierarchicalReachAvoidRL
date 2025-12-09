#!/usr/bin/env python3
"""
Quick utility to measure how far the GO2 robot travels over N high-level steps.
Allows you to keep the existing high/low-level frequency split while inspecting
the displacement statistics to guide horizon choices.
"""

import argparse
import sys
from typing import Tuple

import isaacgym  # noqa: F401 must precede torch import
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args


def parse_custom_args() -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(description="Measure displacement across high-level steps.")
    parser.add_argument("--steps", type=int, default=100, help="Number of consecutive high-level steps to simulate.")
    parser.add_argument("--vx", type=float, default=1.0, help="High-level forward velocity command.")
    parser.add_argument("--vy", type=float, default=0.0, help="High-level lateral velocity command.")
    parser.add_argument("--vyaw", type=float, default=0.0, help="High-level yaw rate command.")
    parser.add_argument(
        "--num-envs", type=int, default=16, help="Number of parallel environments for the test (reduces GPU load)."
    )
    return parser.parse_known_args()


def main() -> None:
    custom_args, remaining = parse_custom_args()
    sys.argv = [sys.argv[0]] + remaining

    args = get_args()
    args.headless = True

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()
    env_cfg.env.num_envs = custom_args.num_envs
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    env_cfg.env.num_envs = custom_args.num_envs

    device = torch.device(getattr(args, "rl_device", "cuda:0"))

    env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )

    obs, g_vals, h_vals = env.reset()
    del obs, g_vals, h_vals  # unused in this diagnostic

    step_action = torch.tensor(
        [custom_args.vx, custom_args.vy, custom_args.vyaw], device=device, dtype=torch.float
    ).unsqueeze(0).repeat(env.num_envs, 1)

    start_pos = env.base_env.base_pos[:, :2].clone()
    print(
        f"Running {custom_args.steps} steps with action "
        f"[vx={custom_args.vx:.2f}, vy={custom_args.vy:.2f}, vyaw={custom_args.vyaw:.2f}] "
        f"using {env.num_envs} envs..."
    )

    for step_idx in range(custom_args.steps):
        env.step(step_action)
        current_pos = env.base_env.base_pos[:, :2]
        displacement = torch.norm(current_pos - start_pos, dim=1)
        disp_cpu = displacement.detach().cpu()
        print(
            f"step {step_idx + 1:02d}: "
            f"mean={disp_cpu.mean():.3f} m | "
            f"min={disp_cpu.min():.3f} m | "
            f"max={disp_cpu.max():.3f} m"
        )

    final_disp = torch.norm(env.base_env.base_pos[:, :2] - start_pos, dim=1).detach().cpu()
    print(
        f"\nAfter {custom_args.steps} steps: "
        f"mean displacement={final_disp.mean():.3f} m | "
        f"median={final_disp.median():.3f} m | "
        f"max={final_disp.max():.3f} m"
    )

    env.close()


if __name__ == "__main__":
    main()
