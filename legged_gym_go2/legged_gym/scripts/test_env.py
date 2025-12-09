import isaacgym
import torch
import time

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def test_env(args):
    """
    Test script to initialize an environment, run it for a few steps with random actions,
    and print out rewards, extras, and agent positions.
    """
    # Create the environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Turn off domain randomization for testing to get a cleaner test
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    print(f"--- Testing environment '{args.task}' ---")
    print("You should see red (unsafe) and green (target) spheres in the simulator.")

    obs, _ = env.reset() # Reset the environment to get initial observations
    
    # Simulation loop
    for i in range(200): # 增加循环次数，方便观察
        # Generate random actions for each environment
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        
        # Step the environment, now unpacking 7 values
        obs, _, rews, dones, infos, avoid, reach = env.step(actions)
        
        # Get agent positions (we only need to look at the first agent, env 0)
        agent_position = env.root_states[0, :3]
        
        # Print information for the first agent (env 0)
        if i % 20 == 0: # 每20步打印一次信息，避免刷屏
            print(f"\n--- Step {i+1} ---")
            print(f"  Agent 0 Position: {agent_position.cpu().numpy()}")
            print(f"  Reward (Agent 0): {rews[0].item():.4f}")
            
            # Print our new, direct metrics
            print(f"  Metrics (Agent 0):")
            print(f"    -> avoid: {avoid[0].item():.4f} (should be > 0 only when inside red sphere)")
            print(f"    -> reach: {reach[0].item():.4f} (distance to green sphere's center)")

            # Print other extras from the infos dict
            if infos.get('episode'):
                print(f"  Extras (Agent 0): {infos.get('episode', {})}")

        # Handle rendering and custom drawing
        if not args.headless:
            # Step graphics
            env.gym.step_graphics(env.sim)
            # Render the main viewer
            env.gym.draw_viewer(env.viewer, env.sim, True)
            # Call our custom drawing function for debug spheres
            env._draw_debug_visualization()
            # Add a small delay
            time.sleep(0.02)

    print("--- Testing finished ---")

if __name__ == '__main__':
    # Use the same argument parser as in train.py
    args = get_args()
    # Force task to 'go2' and run in non-headless mode for visualization
    args.task = "go2"
    args.headless = True
    test_env(args)