### 🏎️ CarRacing-v3 Reinforcement Learning with Custom Wind Dynamics

This project trains a reinforcement learning (RL) agent to efficiently navigate around a closed-loop track in the `CarRacing-v3` environment using the PPO algorithm from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The environment is enhanced with two key customizations:

1. **Grayscale Frame Processing**: Each frame is converted to grayscale and stacked to provide temporal context, reducing the input complexity while preserving essential visual information.
2. **Custom Wind Wrapper**: A custom `WindWrapper` was introduced to simulate lateral wind forces. This wrapper randomly perturbs the steering component of the action vector, mimicking unpredictable crosswinds that make the driving task more dynamic and challenging.

The agent was trained using over 750,000 timesteps and evaluated for performance stability across multiple episodes. Model checkpoints and video recordings are also available for visualization and reproducibility.
