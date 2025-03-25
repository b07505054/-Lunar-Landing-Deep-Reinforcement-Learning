
# -Lunar Lander Policy Gradient

## Overview
This project implements a **Policy Gradient** reinforcement learning algorithm to train an agent to land a spacecraft safely in the *LunarLander-v2* environment from OpenAI Gym. The agent learns to navigate and land between two flags by optimizing a neural network policy using cumulative decaying rewards.

## Features
- **Environment**: LunarLander-v2 (Gym v0.18.3)
- **Algorithm**: Policy Gradient with a 3-layer neural network (8-16-16-4 architecture)
- **Reward Mechanism**: 
  - Original: Immediate rewards
  - Enhanced: Cumulative decaying rewards with γ = 0.99 (`r_t + 0.99 * r_{t+1} + 0.99^2 * r_{t+2} + ...`)
- **Training**: 400 batches, 5 episodes per batch
- **Evaluation**: Average reward over 5 test runs

## Requirements
- Python 3.x
- Libraries:
  ```bash
  pip install gym[box2d]==0.18.3 pyvirtualdisplay tqdm numpy==1.19.5 torch==1.8.1
  ```
- Additional setup for rendering (Linux):
  ```bash
  apt update
  apt install python-opengl xvfb -y
  ```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/lunar-lander-policy-gradient.git
   cd lunar-lander-policy-gradient
   ```
2. **Run the Code**:
   - Open `hw12_reinforcement_learning_english_version.ipynb` in Jupyter Notebook or Google Colab.
   - Execute all cells to train the agent and generate `Action_List.npy`.
3. **Output**:
   - Training plots: `Total Rewards` and `Final Rewards`
   - Test result: Average reward over 5 runs
   - Saved file: `Action_List.npy` (submit this for evaluation)

## Results
- The agent improves over 400 batches, aiming for a solved threshold of 200 points.
- Training progress is visualized with total and final reward plots.

## Acknowledgments
- Built with PyTorch 1.8.1 and Gym 0.18.3.
- Inspired by reinforcement learning lectures from [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf) and others.

