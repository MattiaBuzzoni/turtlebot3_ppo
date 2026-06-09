# TurtleBot3 Navigation with PPO

<video width="700" controls>
  <source src="https://raw.githubusercontent.com/USER/REPO/main/tests/ppo_1000.mp4" type="video/mp4">
</video>

## Overview
 
This project implements a **navigation** system for the TurtleBot3 Burger mobile robot using **Deep Reinforcement Learning**. The agent learns to reach a target position in open space by processing relative goal information, without any prior map of the environment.
 
The PPO algorithm is implemented **from scratch** in PyTorch, giving full control over the training loop, rollout buffer, and policy update logic. The simulation runs in **PyBullet**, with a custom Gymnasium-compatible environment wrapping the robot dynamics.
 
The architecture is designed to be modular and extensible: adding walls and obstacles (LiDAR is already implemented) , or curriculum learning is a natural next step.

---


## Background & Motivation

This project was developed as a hands-on implementation of **Proximal Policy Optimization (PPO)** applied to a simulated robotic navigation task using PyBullet and TurtleBot3.
The focus was on understanding the full RL loop — environment, reward design, and training dynamics — in a setting simple enough to iterate quickly, without adding unnecessary complexity like obstacles or multi-agent interactions.

---


## Architecture
 
```
turtlebot3_ppo/
├── src/
│   ├── cli/            # CLI entry point (train, eval, visualize)
│   ├── gym/            # TurtleBot3Env — Gymnasium wrapper over PyBullet
│   ├── agent/          # PPO agent (actor-critic networks, rollout buffer)
|   ├── playground/     # Playground for testing the TurtleBot3 with different controllers.
│   └── utils/          
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

 
## Observation & Action Space
 
| Component | Description |
|---|---|
| LiDAR scans | N ray distances (normalized), representing proximity to obstacles |
| Goal distance | Euclidean distance to target (normalized) |
| Goal heading | Angle between robot orientation and target direction |
| **Total observation** | `[lidar_rays..., distance, heading]` |
 
| Action | Range | Description |
|---|---|---|
| Linear velocity | `[0, v_max]` | Forward speed |
| Angular velocity | `[-ω_max, ω_max]` | Rotation rate |
 
---

 
## Reward Function
 
The reward signal couples **goal progress** and **heading alignment** into a single multiplicative term, so the agent is rewarded only when it moves *toward* the target *and* faces it correctly.
 
### Definitions
 
Let $\mathbf{p}_t = (x_t, y_t)$ be the robot's base position at step $t$, $\psi_t$ its yaw angle, and $\mathbf{g} = (g_x, g_y)$ the target position.
 
The **goal distance rate** (signed progress along the robot's path) is:
 
$$\Delta d_t = d_{t-1} - d_t, \qquad d_t = \|\mathbf{g} - \mathbf{p}_t\|_2$$
 
The **goal heading error** is the angle between the robot's orientation and the direction to the target:
 
$$G_h = \text{atan2}(g_y - y_t,\ g_x - x_t) - \psi_t$$
 
An alignment coefficient $A$ wraps $G_h$ into $[0, 2\pi)$ via:
 
$$A = \left(0.5 \cdot (G_h + \pi)\right) \bmod 2\pi$$
 
### Reward cases
 
$$r_t =
\begin{cases}
-10 & \text{if } \Delta d_t > 0.5 \text{ or } \Delta d_t \leq 0 \\
200 \cdot (100\,\Delta d_t) \cdot \left(1 - 4\,\left|0.5 - \left(\dfrac{A}{\pi} \bmod 1\right)\right|\right)
& \text{if } 0 < \Delta d_t \leq 0.5
\end{cases}$$
 
### Intuition
 
The inner factor $\left(1 - 4\left|0.5 - \frac{A}{\pi} \bmod 1\right|\right)$ is a **triangular wave** over the heading error, peaking at $+1$ when the robot faces the target directly ($G_h = 0$) and reaching $-1$ when it faces directly away. Combined with the scaled progress $100\,\Delta d_t \in (0, 50]$, the reward is:
 
- **positive and large** when the robot moves fast toward the target while aligned
- **zero** at $90°$ misalignment regardless of speed
- **negative** when moving backward or misaligned, or when $\Delta d_t$ falls outside the valid range

---

 
## Roadmap
 
- [ ] Add static obstacles to the environment
- [ ] Add wall boundaries and confined arena
- [ ] Curriculum learning: start with easy configurations, progressively increase difficulty
- [ ] Multi-goal episodes
      
---


## References

 - Taheri, Hamid, Seyed Rasoul Hosseini, and Mohammad Ali Nekoui. "Deep reinforcement learning with enhanced ppo for safe mobile robot navigation." arXiv preprint arXiv:2405.16266 (2024)
 - Cheng, Ni, Zhong, Wei. "Autonomous Robot Goal Seeking and Collision Avoidance in the Physical World: An Automated Learning and Evaluation Framework Based on the PPO Method." Appl. Sci. 2024,  14, 11020. https://doi.org/10.3390/app142311020 

---

 
## License
 
MIT — see [`LICENSE`](LICENSE).

 
---


