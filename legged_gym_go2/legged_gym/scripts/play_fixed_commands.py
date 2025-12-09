import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play_fixed_commands(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    # 避免命令被周期性重采样与随机heading影响：
    env_cfg.commands.resampling_time = 1e9  # 设为极大，几乎不再重采样
    env_cfg.commands.heading_command = False  # 直接使用我们设置的yaw速度指令

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 定义测试命令序列
    test_commands = [
        [0.0, 0.0, 0.0],   # 停止
        [1.0, 0.0, 0.0],   # 前进
        [0.0, 1.0, 0.0],   # 右移
        [0.0, 0.0, 1.0],   # 右转
        [1.0, 0.0, 0.5],   # 前进+右转
        [0.5, 0.5, 0.0],   # 前进+右移
    ]
    
    command_duration = 200  # 每个命令持续200步
    current_command_idx = 0
    step_count = 0
    
    print("=== GO2 机器狗固定命令测试 ===")
    print("将依次执行以下命令:")
    for i, cmd in enumerate(test_commands):
        print(f"  {i+1}. 前进={cmd[0]:.1f}, 左右={cmd[1]:.1f}, 转向={cmd[2]:.1f}")
    print("=" * 40)

    for i in range(10*int(env.max_episode_length)):
        # 更新命令
        if step_count % command_duration == 0:
            current_command_idx = (current_command_idx + 1) % len(test_commands)
            current_command = test_commands[current_command_idx]
            print(f"切换到命令 {current_command_idx + 1}: 前进={current_command[0]:.1f}, 左右={current_command[1]:.1f}, 转向={current_command[2]:.1f}")
        
        # 设置当前命令
        current_command = test_commands[current_command_idx]
        env.commands[:, 0] = current_command[0]  # 前进速度
        env.commands[:, 1] = current_command[1]  # 左右速度  
        env.commands[:, 2] = current_command[2]  # 转向速度
        
        # 执行动作
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 显示状态
        if step_count % 50 == 0:  # 每50步显示一次
            base_vel = env.base_lin_vel[0].cpu().numpy()
            base_ang_vel = env.base_ang_vel[0].cpu().numpy()
            print(f"实际速度: 前进={base_vel[0]:.2f}, 左右={base_vel[1]:.2f}, 转向={base_ang_vel[2]:.2f}")
        
        # 如果环境重置，重新设置命令
        if dones[0]:
            env.commands[:, 0] = current_command[0]
            env.commands[:, 1] = current_command[1] 
            env.commands[:, 2] = current_command[2]
        
        step_count += 1

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play_fixed_commands(args)
