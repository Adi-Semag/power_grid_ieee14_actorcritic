# Power Grid Reinforcement Learning

This project implements a reinforcement learning approach for power grid control using an Actor-Critic algorithm.

## Project Structure

- `actor_critic.py`: Implementation of the Actor-Critic reinforcement learning algorithm
- `power_grid_env.py`: Power grid environment simulation based on the IEEE 14-bus system
- `train.py`: Main script to train the reinforcement learning agent
- `requirements.txt`: Required dependencies for the project

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (optional but recommended):

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Train the RL agent:

```bash
python train.py
```

This will start the training process with the default parameters. The training progress will be displayed in the console, and the results will be saved in a timestamped directory (e.g., `results_20240601_120000`).

## Training Results

The training script will:
- Display training progress including rewards, losses, and evaluation metrics
- Save the best model as `best_model.pth` in the results directory
- Create checkpoints every 100 episodes
- Generate training visualization plots at the end of training

## Customization

You can modify the training parameters in `train.py`:
- `num_episodes`: Number of episodes to train (default: 3000)
- `max_steps`: Maximum steps per episode (default: 300)
- `eval_interval`: How often to evaluate the agent (default: 10 episodes)
- `patience`: Early stopping threshold (default: 100)

The agent parameters can be configured in the ActorCritic initialization in `train.py`:
- Learning rates
- Hidden dimension size
- Discount factor (gamma)
- Replay buffer size
- Exploration parameters (epsilon)

## Environment

The power grid environment simulates an IEEE 14-bus system with voltage control, active power, and reactive power components. The environment implements curriculum learning to gradually increase the complexity of the control task. 
https://labs.ece.uw.edu/pstca/pf14/ieee14cdf.txt