# HD-MCRA: Hierarchical Minimum-Cost Reach-Avoid Learning for Safe Quadruped Navigation

[Project website](https://haoyangc2001.github.io/Go2HierarchicalReachAvoidRL/)

HD-MCRA studies budget-aware, safe navigation for quadruped robots in complex environments. The project formulates navigation as a **Minimum-Cost Reach-Avoid (MCRA)** problem: reach the goal, avoid unsafe states throughout the trajectory, and minimize cumulative execution cost under a remaining budget.

## Key Ideas

- **Trajectory-level MCRA formulation.** An augmented state represents task progress, safety evolution, and remaining budget together, treating reach-avoid as a hard constraint rather than a reward-weighted preference.
- **Dual critics and budget-conditioned policy optimization.** A reachability critic estimates safe feasibility while a cost critic estimates cumulative execution cost. The high-level policy prioritizes cost reduction when feasible and feasibility recovery when safety or budget is at risk.
- **Hierarchical closed-loop control.** A robust frozen locomotion policy tracks low-dimensional velocity commands, allowing the high-level navigator to operate in a stable closed-loop action space instead of directly controlling high-dimensional joints.
- **Data-driven robust margins.** Safety and cost margins, estimated from closed-loop data, compensate for trajectory deviations, cost overruns, and sim-to-real mismatch.

## Results

In difficult Isaac Gym navigation environments, HD-MCRA improves navigation success by approximately **20%** and reduces average execution cost by approximately **30%** compared with conventional baselines. The system has also been demonstrated on an AGIROS quadruped robot.

## Repository Layout

```text
legged_gym_go2/   Go2 environments and hierarchical navigation implementation
rsl_rl/           Reinforcement learning components
isaacgym/          Isaac Gym dependencies and assets
unitree_rl_gym/    Unitree robot training utilities
webui/             Static project website for GitHub Pages
ALGORITHM_DESIGN.md  Algorithm design notes
```

## Status

This repository contains the project implementation and its accompanying research website. Training and deployment requirements depend on the local Isaac Gym, CUDA, and robot software environment; release details will be updated as the project evolves.

## Acknowledgements

This project builds on the open-source ecosystems of [legged_gym_go2](https://github.com/littlebearqqq/legged_gym_go2), [rsl_rl](https://github.com/leggedrobotics/rsl_rl), Isaac Gym, and Unitree robot software.

## Citation

Citation information will be added after publication.
