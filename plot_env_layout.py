#!/usr/bin/env python3
"""
Visualize GO2 high-level environment layout (target + obstacles).
"""

import matplotlib.pyplot as plt

# obstacle/target config matches GO2HighLevelCfg.rewards_ext
TARGET_POS = (0.0, 0.0)
TARGET_RADIUS = 1.0
OBSTACLE_POSITIONS = [
    (-2.66, -3.66),
    (4.00, -3.00),
    (-3.66, 2.33),
    (3.00, 4.33),
    (0.33, -4.00),
    (-4.33, -0.33),
    (2.66, 0.33),
    (-0.66, 3.33),
]
OBSTACLE_RADIUS = 0.67


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("GO2 Reach-Avoid Layout (Top View)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Draw obstacles
    for idx, (x, y) in enumerate(OBSTACLE_POSITIONS, start=1):
        circle = plt.Circle((x, y), OBSTACLE_RADIUS, color="red", alpha=0.35, edgecolor="darkred")
        ax.add_patch(circle)
        ax.text(x, y, f"H{idx}", color="darkred", ha="center", va="center", fontsize=8)

    # Draw target
    target_circle = plt.Circle(TARGET_POS, TARGET_RADIUS, color="green", alpha=0.35, edgecolor="darkgreen")
    ax.add_patch(target_circle)
    ax.text(
        TARGET_POS[0],
        TARGET_POS[1],
        "Target",
        color="darkgreen",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
