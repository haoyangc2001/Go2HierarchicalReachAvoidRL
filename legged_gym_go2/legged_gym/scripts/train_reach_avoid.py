#!/usr/bin/env python3
import os
import time
from datetime import datetime
from typing import Tuple

import isaacgym
import torch

from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args

from rsl_rl.algorithms.reach_avoid_ppo import ReachAvoidPPO
from rsl_rl.modules import ActorCritic


class HierarchicalVecEnv:
    def __init__(self, env: HierarchicalGO2Env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.device = env.device

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, g_vals, h_vals = self.env.reset()
        return obs, g_vals, h_vals

    def step(self, actions: torch.Tensor):
        obs, g_vals, h_vals, dones, infos = self.env.step(actions)
        return obs, g_vals, h_vals, dones, infos

    def close(self) -> None:
        self.env.close()


def create_env(env_cfg, train_cfg, args, device) -> HierarchicalVecEnv:
    base_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )
    return HierarchicalVecEnv(base_env)


def compute_reach_avoid_success_rate(g_sequence: torch.Tensor, h_sequence: torch.Tensor) -> float:
    with torch.no_grad():
        time_steps, num_envs = g_sequence.shape
        g_negative = g_sequence < 0
        has_success = g_negative.any(dim=0)
        first_success = torch.argmax(g_negative.long(), dim=0)
        first_indices = torch.where(
            has_success,
            first_success,
            torch.full((num_envs,), time_steps, device=g_sequence.device, dtype=torch.long),
        )

        time_index = torch.arange(time_steps, device=g_sequence.device).unsqueeze(1)
        before_success = time_index < first_indices.unsqueeze(0)

        h_violation = (h_sequence >= 0) & before_success
        safe_before = ~h_violation.any(dim=0)

        success = has_success & safe_before
        return success.float().mean().item()


def train_reach_avoid(args) -> None:
    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()

    # 配置网络大小
    train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]
    train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]

    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    device = torch.device(args.rl_device)

    env = create_env(env_cfg, train_cfg, args, device)

    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std,
    ).to(device)

    alg = ReachAvoidPPO(
        actor_critic=actor_critic,
        device=device,
        **train_cfg.algorithm.__dict__,
    )
    alg.init_storage(
        num_envs=env.num_envs,
        horizon=train_cfg.algorithm.num_steps_per_env,
        obs_shape=(env.num_obs,),
        action_shape=(env.num_actions,),
    )

    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", train_cfg.runner.experiment_name, log_timestamp)
    start_iteration = 0
    if getattr(train_cfg.runner, "resume", False):
        resume_path = getattr(train_cfg.runner, "resume_path", "")
        if resume_path and os.path.isfile(resume_path):
            print(f"resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            actor_state = checkpoint.get("actor_critic")
            if actor_state is not None:
                actor_critic.load_state_dict(actor_state)
            opt_state = checkpoint.get("optimizer")
            if opt_state is not None:
                alg.optimizer.load_state_dict(opt_state)
            start_iteration = checkpoint.get("iteration", 0)
            prev_log_dir = os.path.dirname(resume_path)
            print(f"  continuing from iteration {start_iteration}")
            print(f"  previous checkpoints were in: {prev_log_dir}")
            print(f"  new checkpoints will be saved to: {log_dir}")
        else:
            print(f"warning: resume enabled but checkpoint not found: {resume_path}")
            print("         starting a new run instead.")
    os.makedirs(log_dir, exist_ok=True)
    gh_dump_interval = getattr(train_cfg.runner, "gh_dump_interval", 0)
    gh_dump_dir = None
    if gh_dump_interval and gh_dump_interval > 0:
        gh_dump_dir = os.path.join(log_dir, "gh_snapshots")
        os.makedirs(gh_dump_dir, exist_ok=True)

    print("Reach-Avoid training")
    print(f"  envs       : {env.num_envs}")
    print(f"  obs dim    : {env.num_obs}")
    print(f"  action dim : {env.num_actions}")
    print(f"  horizon    : {train_cfg.algorithm.num_steps_per_env}")
    print(f"  device     : {device}")
    print(f"  log dir    : {log_dir}")

    obs, g_vals, h_vals = env.reset()
    obs = obs.to(device)
    g_vals = g_vals.to(device)
    h_vals = h_vals.to(device)
    horizon = train_cfg.algorithm.num_steps_per_env

    max_iterations = train_cfg.runner.max_iterations
    save_interval = train_cfg.runner.save_interval
    success_rate = 0.0
    interval_start = time.time()

    for iteration in range(start_iteration, max_iterations):
        rollout_obs = obs.new_empty(horizon + 1, env.num_envs, env.num_obs)
        rollout_actions = obs.new_empty(horizon, env.num_envs, env.num_actions)
        rollout_log_probs = obs.new_empty(horizon, env.num_envs)
        rollout_values = obs.new_empty(horizon, env.num_envs)
        rollout_g = g_vals.new_empty(horizon + 1, env.num_envs)
        rollout_h = h_vals.new_empty(horizon + 1, env.num_envs)
        rollout_dones = torch.empty(horizon, env.num_envs, device=device, dtype=torch.bool)

        rollout_obs[0].copy_(obs)
        rollout_g[0].copy_(g_vals)
        rollout_h[0].copy_(h_vals)

        for step in range(horizon):
            actions, log_probs, values = alg.act(rollout_obs[step])
            next_obs, next_g, next_h, dones, _ = env.step(actions)

            next_obs = next_obs.to(device)
            next_g = next_g.to(device)
            next_h = next_h.to(device)
            dones = dones.to(device)

            rollout_actions[step].copy_(actions)
            rollout_log_probs[step].copy_(log_probs)
            rollout_values[step].copy_(values)
            rollout_dones[step].copy_(dones.bool())

            rollout_obs[step + 1].copy_(next_obs)
            rollout_g[step + 1].copy_(next_g)
            rollout_h[step + 1].copy_(next_h)

            obs = next_obs
            g_vals = next_g
            h_vals = next_h

        # force horizon truncation to behave like episode termination
        rollout_dones[-1].fill_(True)

        alg.buffer.store_rollout(
            observations=rollout_obs,
            actions=rollout_actions,
            log_probs=rollout_log_probs,
            values=rollout_values,
            g_values=rollout_g,
            h_values=rollout_h,
            dones=rollout_dones,
        )

        with torch.no_grad():
            last_values = alg.actor_critic.evaluate(obs).squeeze(-1)

        alg.buffer.compute_advantages(last_values, alg.gamma, alg.lam)
        success_rate = compute_reach_avoid_success_rate(
            alg.buffer.g_values[1:], alg.buffer.h_values[1:]
        )
        policy_loss, value_loss = alg.update()
        value_stats = getattr(alg, "last_value_stats", {})
        v_mean = value_stats.get("value_mean", float("nan"))
        r_mean = value_stats.get("return_mean", float("nan"))
        v_rmse = value_stats.get("value_rmse", float("nan"))
        v_expvar = value_stats.get("explained_variance", float("nan"))
        adv_std = value_stats.get("adv_std", float("nan"))

        if (iteration + 1) % 1 == 0:
            elapsed = time.time() - interval_start
            print(
                f"iter {iteration + 1:05d} | success {success_rate:.3f} | "
                f"policy_loss {policy_loss:.5f} | value_loss {value_loss:.5f} | Vmean {v_mean:.3f} | Rmean {r_mean:.3f} | Vrmse {v_rmse:.3f} | VexpVar {v_expvar:.3f} | adv_std {adv_std:.3f} | elapsed {elapsed:.2f}s"
            )
            interval_start = time.time()

        if (iteration + 1) % save_interval == 0:
            save_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
            torch.save(
                {
                    "actor_critic": alg.actor_critic.state_dict(),
                    "optimizer": alg.optimizer.state_dict(),
                    "iteration": iteration + 1,
                    "success_rate": success_rate,
                    "low_level_model_path": train_cfg.runner.low_level_model_path,
                },
                save_path,
            )
            print(f"  saved checkpoint: {save_path}")

        # start the next rollout from a freshly reset environment
        if iteration + 1 < max_iterations:
            obs, g_vals, h_vals = env.reset()
            obs = obs.to(device)
            g_vals = g_vals.to(device)
            h_vals = h_vals.to(device)

    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(
        {
            "actor_critic": alg.actor_critic.state_dict(),
            "optimizer": alg.optimizer.state_dict(),
            "iteration": max_iterations,
            "success_rate": success_rate,
            "low_level_model_path": train_cfg.runner.low_level_model_path,
        },
        final_path,
    )
    print(f"training complete. final checkpoint: {final_path}")

    env.close()


if __name__ == "__main__":
    args = get_args()
    args.headless = True
    args.compute_device_id = 1
    args.sim_device_id = 1
    args.rl_device = "cuda:1"
    args.sim_device = "cuda:1"
    train_reach_avoid(args)



