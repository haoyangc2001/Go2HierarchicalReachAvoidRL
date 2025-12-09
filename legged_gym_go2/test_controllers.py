#!/usr/bin/env python3
"""
测试顶层和底层控制器的集成效果
基于train_high_level.py的方式创建环境并运行几步
"""

import os
import numpy as np
import isaacgym
from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.utils import get_args
from rsl_rl.modules import ActorCritic
import torch


def test_controllers():
    """测试控制器集成效果"""
    print("=" * 80)
    print("测试顶层和底层控制器集成效果")
    print("=" * 80)
    
    # 使用与训练代码相同的参数设置
    args = get_args()
    args.headless = True
    args.num_envs = 1
    args.rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.sim_device = args.rl_device
    
    print(f"使用设备: {args.rl_device}")
    
    # 加载配置
    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()
    
    print(f"\n1. 检查模型文件...")
    
    # 检查底层策略模型
    low_level_path = train_cfg.runner.low_level_model_path
    if os.path.exists(low_level_path):
        print(f"   ✓ 底层策略模型存在: {low_level_path}")
    else:
        print(f"   ✗ 底层策略模型不存在: {low_level_path}")
        return
    
    # 检查高层模型
    high_level_path = "logs/high_level_go2/model_2000.pt"
    if os.path.exists(high_level_path):
        print(f"   ✓ 高层模型存在: {high_level_path}")
    else:
        print(f"   ✗ 高层模型不存在: {high_level_path}")
        return
    
    print(f"\n2. 创建分层环境...")
    try:
        # 创建分层环境
        hierarchical_env = HierarchicalGO2Env(
            cfg=env_cfg,
            low_level_model_path=low_level_path,
            args=args,
            device=args.rl_device
        )
        print(f"   ✓ 分层环境创建成功")
        print(f"   - 环境数量: {hierarchical_env.num_envs}")
        print(f"   - 高层观测维度: {hierarchical_env.num_obs}")
        print(f"   - 高层动作维度: {hierarchical_env.num_actions}")
    except Exception as e:
        print(f"   ✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n3. 加载高层模型...")
    try:
        # 创建高层网络
        actor_critic = ActorCritic(
            num_actor_obs=hierarchical_env.num_obs,
            num_critic_obs=hierarchical_env.num_obs,
            num_actions=hierarchical_env.num_actions,
            actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
            activation=train_cfg.policy.activation,
            init_noise_std=train_cfg.policy.init_noise_std
        ).to(args.rl_device)
        
        # 加载模型
        checkpoint = torch.load(high_level_path, map_location=args.rl_device)
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        actor_critic.eval()
        
        print(f"   ✓ 高层模型加载成功")
        print(f"   - 训练迭代: {checkpoint.get('iteration', 'unknown')}")
        if 'infos' in checkpoint:
            infos = checkpoint['infos']
            print(f"   - 平均g值: {infos.get('mean_g_value', 'N/A')}")
            print(f"   - 平均h值: {infos.get('mean_h_value', 'N/A')}")
    except Exception as e:
        print(f"   ✗ 高层模型加载失败: {e}")
        hierarchical_env.close()
        return
    
    print(f"\n4. 运行控制器测试...")
    try:
        # 重置环境
        obs, g_values, h_values = hierarchical_env.reset()
        
        print(f"   初始状态:")
        print(f"     - 观测形状: {obs.shape}")
        print(f"     - 平均g值: {g_values.mean().item():.3f}")
        print(f"     - 平均h值: {h_values.mean().item():.3f}")
        print(f"     - 观测范围: [{obs.min().item():.3f}, {obs.max().item():.3f}]")
        
        # 运行几步测试
        num_steps = 100
        print(f"\n   运行 {num_steps} 步测试:")
        
        for step in range(num_steps):
            # 获取高层动作
            with torch.no_grad():
                high_level_actions = actor_critic.act_inference(obs)
            
            # 执行动作
            obs, g_values, h_values, dones, infos = hierarchical_env.step(high_level_actions)
            
            # 期望速度（由高层动作经过缩放得到）
            with torch.no_grad():
                clipped = torch.clamp(high_level_actions, -1.0, 1.0)
                action_scale = hierarchical_env.high_level_env.action_scale
                desired_cmds = clipped * action_scale  # [vx, vy, vyaw]
            
            # 实际速度（由底层控制器执行后的机体速度）
            actual_lin_vel = hierarchical_env.base_env.base_lin_vel  # [N, 3]
            actual_ang_vel = hierarchical_env.base_env.base_ang_vel  # [N, 3]
            
            # 仅打印第一个环境（通常num_envs=1，更直观）
            vx_des, vy_des, wz_des = desired_cmds[0, 0].item(), desired_cmds[0, 1].item(), desired_cmds[0, 2].item()
            vx_act, vy_act, wz_act = actual_lin_vel[0, 0].item(), actual_lin_vel[0, 1].item(), actual_ang_vel[0, 2].item()
            
            # 输出每步信息
            if step % 5 == 0:
                print(f"     步骤 {step}:")
                print(f"       - 高层原始动作(均值): {high_level_actions.mean(dim=0).cpu().numpy()}")
                print(f"       - 期望速度[vx, vy, wz]: [{vx_des:.3f}, {vy_des:.3f}, {wz_des:.3f}]")
                print(f"       - 实际速度[vx, vy, wz]: [{vx_act:.3f}, {vy_act:.3f}, {wz_act:.3f}]")
                print(f"       - 平均g值: {g_values.mean().item():.3f}")
                print(f"       - 平均h值: {h_values.mean().item():.3f}")
                print(f"       - 终止率: {dones.float().mean().item():.3f}")
                
                if 'avoid_metric' in infos:
                    print(f"       - 避障指标: {infos['avoid_metric'].mean().item():.3f}")
                if 'reach_metric' in infos:
                    print(f"       - 到达指标: {infos['reach_metric'].mean().item():.3f}")
            
            # 检查终止条件
            if dones.any():
                reset_obs, reset_g, reset_h = hierarchical_env.reset()
                obs[dones] = reset_obs[dones]
                g_values[dones] = reset_g[dones]
                h_values[dones] = reset_h[dones]
        
        # 分析结果
        print(f"\n5. 测试结果分析:")
        print(f"     - 最终平均g值: {g_values.mean().item():.3f}")
        print(f"     - 最终平均h值: {h_values.mean().item():.3f}")
        print(f"     - 高层动作范围: [{high_level_actions.min().item():.3f}, {high_level_actions.max().item():.3f}]")
        print(f"     - 高层动作均值: {high_level_actions.mean(dim=0).cpu().numpy()}")
        print(f"     - 高层动作标准差: {high_level_actions.std(dim=0).cpu().numpy()}")
        
        # 判断控制器质量
        print(f"\n6. 控制器质量评估:")
        
        # 检查g值是否合理
        if g_values.mean().item() < -100:
            print("     ✓ g值良好，目标函数性能正常")
        elif g_values.mean().item() < 0:
            print("     ⚠ g值一般，目标函数性能有限")
        else:
            print("     ✗ g值较差，目标函数性能差")
        
        # 检查h值是否合理
        if h_values.mean().item() < -100:
            print("     ✓ h值良好，安全函数性能正常")
        elif h_values.mean().item() < 0:
            print("     ⚠ h值一般，安全函数性能有限")
        else:
            print("     ✗ h值较差，安全函数性能差")
        
        # 检查动作多样性
        action_std = high_level_actions.std(dim=0).mean().item()
        if action_std > 0.1:
            print("     ✓ 动作多样性良好，策略探索充分")
        elif action_std > 0.05:
            print("     ⚠ 动作多样性一般，策略探索有限")
        else:
            print("     ✗ 动作多样性不足，策略可能过于保守")
        
        # 检查动作范围
        action_range = [high_level_actions.min().item(), high_level_actions.max().item()]
        if 0.1 < abs(action_range[0]) < 0.8 and 0.1 < abs(action_range[1]) < 0.8:
            print("     ✓ 动作范围合理")
        else:
            print("     ✗ 动作范围不合理")
        
        print(f"   ✓ 控制器测试完成")
        
    except Exception as e:
        print(f"   ✗ 控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 关闭环境
    hierarchical_env.close()
    
    print(f"\n" + "=" * 80)
    print("控制器测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    test_controllers()
