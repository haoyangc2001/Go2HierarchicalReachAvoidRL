#!/usr/bin/env python3

import os
from datetime import datetime

# Isaac Gym imports - 必须最先导入
import isaacgym
import torch
from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.utils import get_args

# RSL_RL imports
from rsl_rl.algorithms.reach_avoid_ppo import ReachAvoidPPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv


class HierarchicalVecEnv(VecEnv):
    """将分层环境包装为 RSL_RL 中的 VecEnv 接口。"""

    def __init__(self, hierarchical_env: HierarchicalGO2Env):
        self.hierarchical_env = hierarchical_env

        # 环境属性
        self.num_envs = hierarchical_env.num_envs
        self.num_obs = hierarchical_env.num_obs
        self.num_actions = hierarchical_env.num_actions
        self.device = hierarchical_env.device

        # 观测与奖励缓冲区
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def step(self, actions):
        """执行一步动作。"""
        obs, g_values, h_values, dones, infos = self.hierarchical_env.step(actions)

        self.obs_buf[:] = obs
        self.reset_buf[:] = dones

        return self.obs_buf, g_values, h_values, self.reset_buf, infos

    def reset(self):
        """重置所有环境。"""
        obs, g_values, h_values = self.hierarchical_env.reset()
        self.obs_buf[:] = obs
        self.reset_buf[:] = False

        return self.obs_buf, g_values, h_values

    def get_observations(self):
        """返回当前观测。"""
        return self.obs_buf

    def get_privileged_observations(self):
        """高层策略不使用特权观测。"""
        return None

    def close(self):
        """关闭底层环境。"""
        self.hierarchical_env.close()


def create_hierarchical_env(env_cfg, train_cfg, args, device):
    """创建分层环境并包装为 VecEnv。"""

    hierarchical_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device
    )

    return HierarchicalVecEnv(hierarchical_env)


def compute_reach_avoid_success_rate(g_sequence: torch.Tensor, h_sequence: torch.Tensor) -> float:
    """根据 g/h 序列计算 reach-avoid 成功率。"""
    with torch.no_grad():
        g_sequence = g_sequence.detach()
        h_sequence = h_sequence.detach()
        time_steps = g_sequence.shape[0]
        num_envs = g_sequence.shape[1]

        g_negative = g_sequence < 0
        has_g_negative = g_negative.any(dim=0)

        first_indices = torch.full((num_envs,), time_steps, dtype=torch.long, device=g_sequence.device)
        if has_g_negative.any():
            first_success = torch.argmax(g_negative.float(), dim=0)
            first_indices = torch.where(has_g_negative, first_success, first_indices)

        time_indices = torch.arange(time_steps, device=g_sequence.device).unsqueeze(1)
        before_success_mask = time_indices < first_indices.unsqueeze(0)

        h_violation = (h_sequence >= 0) & before_success_mask
        all_h_negative = ~h_violation.any(dim=0)

        success = has_g_negative & all_h_negative
        return success.float().mean().item()


def train_high_level_policy(args):
    """训练高层导航策略。"""

    print("=" * 50)
    print("开始训练高层导航策略")
    print("=" * 50)

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()

    from legged_gym.utils.helpers import update_cfg_from_args
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    device = args.rl_device

    print("创建分层环境...")
    env = create_hierarchical_env(env_cfg, train_cfg, args, device)

    print("初始化 PPO 算法...")
    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std
    ).to(device)

    alg = ReachAvoidPPO(actor_critic, device=device, **train_cfg.algorithm.__dict__)
    alg.init_storage(
        env.num_envs,
        train_cfg.algorithm.num_steps_per_env,
        [env.num_obs],
        [env.num_obs],
        [env.num_actions]
    )

    max_iterations = train_cfg.runner.max_iterations
    save_interval = train_cfg.runner.save_interval
    start_iteration = 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", train_cfg.runner.experiment_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    if train_cfg.runner.resume:
        resume_path = getattr(train_cfg.runner, 'resume_path', None)
        if resume_path and os.path.exists(resume_path):
            print(f"从 checkpoint 恢复训练: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)

            actor_critic.load_state_dict(checkpoint['model_state_dict'])
            alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_iteration = checkpoint['iteration'] + 1
            print(f"从迭代 {start_iteration} 继续训练")

            if 'infos' in checkpoint:
                infos = checkpoint['infos']
                print(f"  checkpoint 平均 g 值: {infos.get('mean_g_value', 'N/A')}")
                print(f"  checkpoint 平均 h 值: {infos.get('mean_h_value', 'N/A')}")
        else:
            print(f"警告: resume=True 但找不到 checkpoint 文件: {resume_path}")
            print("将从头开始训练")

    print("训练配置:")
    print(f"  - 最大迭代次数: {max_iterations}")
    print(f"  - 环境数量: {env.num_envs}")
    print(f"  - 观测维度: {env.num_obs}")
    print(f"  - 动作维度: {env.num_actions}")
    print(f"  - 设备: {device}")
    print(f"  - 日志目录: {log_dir}")

    obs, g_values, h_values = env.reset()

    for iteration in range(start_iteration, max_iterations):
        print(f"\n--- 迭代 {iteration + 1}/{max_iterations} ---")

        obs_list = [obs]
        g_list = [g_values]
        h_list = [h_values]
        actions_list = []
        values_list = []
        actions_log_prob_list = []
        action_mean_list = []
        action_std_list = []
        dones_list = []
        infos_list = []

        for step in range(train_cfg.algorithm.num_steps_per_env):
            actions = alg.act(obs, obs)

            actions_list.append(actions)
            values_list.append(alg.transition.values.clone())
            actions_log_prob_list.append(alg.transition.actions_log_prob.clone())
            action_mean_list.append(alg.transition.action_mean.clone())
            action_std_list.append(alg.transition.action_std.clone())

            obs, g_values, h_values, dones, infos = env.step(actions)
            obs_list.append(obs)
            g_list.append(g_values)
            h_list.append(h_values)
            dones_list.append(dones)
            infos_list.append(infos)

            if dones.any():
                reset_obs, reset_g, reset_h = env.reset()
                obs[dones] = reset_obs[dones]
                g_values[dones] = reset_g[dones]
                h_values[dones] = reset_h[dones]

        obs_tensor = torch.stack(obs_list[:-1])
        g_tensor = torch.stack(g_list[:-1])
        h_tensor = torch.stack(h_list[:-1])
        actions_tensor = torch.stack(actions_list)
        values_tensor = torch.stack(values_list)
        actions_log_prob_tensor = torch.stack(actions_log_prob_list)
        action_mean_tensor = torch.stack(action_mean_list)
        action_std_tensor = torch.stack(action_std_list)

        alg.process_episode(
            obs_tensor,
            g_tensor,
            h_tensor,
            actions_tensor,
            values_tensor,
            actions_log_prob_tensor,
            action_mean_tensor,
            action_std_tensor,
            dones_list,
            infos_list
        )

        last_obs = obs_list[-1]
        alg.compute_returns(last_obs)

        mean_value_loss, mean_surrogate_loss = alg.update()

        if iteration % 5 == 0:
            g_history = torch.stack(g_list[1:])
            h_history = torch.stack(h_list[1:])
            success_rate = compute_reach_avoid_success_rate(g_history, h_history)
            print(f"  reach-avoid 成功率: {success_rate:.3f}")
            print(f"  value loss: {mean_value_loss:.6f}")
            print(f"  policy loss: {mean_surrogate_loss:.6f}")

        if iteration % save_interval == 0:
            model_path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': alg.optimizer.state_dict(),
                'iteration': iteration,
                'infos': {
                    'mean_g_value': g_values.mean().item(),
                    'mean_h_value': h_values.mean().item(),
                    'mean_value_loss': mean_value_loss,
                    'mean_surrogate_loss': mean_surrogate_loss
                }
            }, model_path)
            print(f"  模型已保存: {model_path}")

    final_model_path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        'model_state_dict': actor_critic.state_dict(),
        'optimizer_state_dict': alg.optimizer.state_dict(),
        'iteration': max_iterations,
        'infos': {
            'mean_g_value': g_values.mean().item(),
            'mean_h_value': h_values.mean().item(),
            'mean_value_loss': mean_value_loss,
            'mean_surrogate_loss': mean_surrogate_loss
        }
    }, final_model_path)

    print(f"\n训练完成，最终模型已保存至: {final_model_path}")

    env.close()


if __name__ == '__main__':
    args = get_args()
    args.headless = True
    args.compute_device_id = 1
    args.sim_device_id = 1
    args.rl_device = "cuda:1"
    args.sim_device = "cuda:1"
    train_high_level_policy(args)