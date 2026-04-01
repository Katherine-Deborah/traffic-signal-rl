# Multi-Agent RL Traffic Signal Control

A reinforcement learning system that learns to control a traffic signal at a single intersection using SUMO (Simulation of Urban MObility) and PyTorch. Two RL algorithms — DQN and PPO — are trained from scratch and benchmarked against a fixed-time baseline across multiple traffic scenarios.

---

## Project Structure

```
traffic/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── env/
│   └── traffic_env.py           # Gymnasium environment wrapping SUMO via traci
├── agents/
│   ├── base_agent.py            # Abstract interface + GPU/CPU selector
│   ├── dqn_agent.py             # DQN: Q-network, replay buffer, epsilon-greedy
│   └── ppo_agent.py             # PPO: actor-critic, rollout buffer, GAE
├── baselines/
│   └── fixed_time.py            # Fixed-cycle signal controller (comparison target)
├── network/
│   ├── generate_network.py      # Generates SUMO .xml files and route files
│   └── single/                  # Generated network files (created by step 2)
│       ├── single.net.xml
│       ├── routes_normal.rou.xml
│       ├── routes_ns_peak.rou.xml
│       ├── routes_ew_peak.rou.xml
│       ├── routes_high.rou.xml
│       └── single.sumocfg
├── training/
│   ├── train.py                 # DQN and PPO training loops with MLflow + TensorBoard
│   └── evaluate.py              # Runs trained agents, compares to baseline, saves plots
├── experiments/
│   ├── reward_ablation.py       # Experiment: which reward function works best?
│   └── state_ablation.py        # Experiment: which state features matter most?
├── results/
│   ├── checkpoints/             # Saved model weights (.pt files)
│   ├── metrics/                 # CSV files with evaluation numbers
│   ├── plots/                   # PNG comparison charts
│   └── tensorboard/             # TensorBoard event files
└── mlruns/                      # MLflow run history
```

---

## Quick Start

Run these commands from the project root (`Documents/traffic/`) in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the SUMO network and route files
python network/generate_network.py

# 3. Train DQN (500 episodes, ~30–60 min)
python training/train.py --algo dqn

# 4. Train PPO (500 episodes, ~30–60 min)
python training/train.py --algo ppo

# 5. Evaluate both agents vs. fixed-time baseline
python training/evaluate.py

# 6. Reward function ablation experiment
python experiments/reward_ablation.py

# 7. State feature ablation experiment
python experiments/state_ablation.py
```

All steps are idempotent — re-running overwrites previous outputs.

---

## The Simulation

**SUMO** simulates a single 4-way intersection with a traffic light named `center`. There are 4 incoming edges (N2C, S2C, E2C, W2C) and 4 outgoing edges. Each edge has 2 lanes. Speed limit is 50 km/h (13.89 m/s). Each edge is 300 m long.

Each simulated episode covers **1 hour of traffic** (3600 simulated seconds). The RL agent makes a decision every 5 seconds (`delta_time = 5`), giving 720 steps per episode.

---

## The Environment (`env/traffic_env.py`)

The environment follows the [Gymnasium](https://gymnasium.farama.org/) API (`reset`, `step`, `close`).

### Observation (State)

At each step, the agent sees a vector built from the following features, all normalized to [0, 1]:

| Feature | Source | Normalization cap |
|---------|--------|-----------------|
| Queue length per edge (×4) | Halting vehicles on each incoming edge | 30 vehicles |
| Density per edge (×4) | Lane occupancy percentage | 100% |
| Waiting time per edge (×4) | Cumulative seconds vehicles have been waiting | 300 s |
| Current phase (one-hot, ×2) | Which green phase is active | — |
| Phase duration (×1) | Seconds in current phase | 120 s |

Default state vector size: **15 values** (4+4+4+2+1).

### Action

`Discrete(2)` — choose which green phase to activate:
- Phase 0: North-South green (N↕S)
- Phase 1: East-West green (E↔W)

**Yellow transitions are inserted automatically** — when the agent switches phases, the environment inserts a 3-second yellow before activating the new green. The agent does not need to handle this.

**Minimum green enforcement** — the agent cannot switch phases until the current phase has been green for at least 10 seconds (`min_green`). If it tries, the action is overridden to keep the current phase.

### Reward

Four options, set via `reward.type` in `config.yaml`:

| Type | Formula | What it optimizes |
|------|---------|-----------------|
| `waiting_time` | `−Σ waiting_time` across all edges | Minimize total seconds vehicles wait |
| `queue_length` | `−Σ queue_length` across all edges | Minimize total vehicles stuck |
| `combined` | `−(0.5 × waiting + 0.5 × queue)` | Weighted combination |
| `pressure` | `−|queue_red_edges − queue_green_edges|` | Serve the most congested direction |

All rewards are scaled by `reward.scale = 0.001` to keep values in a stable range for neural network training.

---

## Traffic Scenarios (Route Files)

Four scenarios are generated by `generate_network.py`, each representing different real-world conditions:

| File | Description | N↕S flow | E↔W flow |
|------|------------|---------|---------|
| `routes_normal.rou.xml` | Balanced traffic, moderate volume | 300 veh/hr | 300 veh/hr |
| `routes_ns_peak.rou.xml` | Morning rush, North-South dominant | 600 veh/hr | 200 veh/hr |
| `routes_ew_peak.rou.xml` | Evening rush, East-West dominant | 200 veh/hr | 600 veh/hr |
| `routes_high.rou.xml` | High congestion, all directions saturated | 800 veh/hr | 800 veh/hr |

To train on a specific scenario:
```bash
python training/train.py --algo dqn --scenario ns_peak
python training/evaluate.py --scenario ew_peak
```

**To add a new scenario:** open `network/generate_network.py`, add an entry to the `scenarios` dict in `write_route_file()`, then re-run `python network/generate_network.py`. The new `.rou.xml` file will appear in `network/single/`. No other files need to change.

---

## The Agents

### DQN (`agents/dqn_agent.py`)

**Deep Q-Network** — an off-policy, value-based method.

How it works:
1. At each step, pick an action using **epsilon-greedy**: with probability ε, explore randomly; otherwise, pick the action with the highest predicted Q-value.
2. Store the experience `(state, action, reward, next_state, done)` in a **replay buffer** (capacity 50,000).
3. Every step, sample a random mini-batch of 64 experiences and update the Q-network using the Bellman equation: `Q(s,a) = r + γ · max Q'(s', a')`.
4. The loss function is **Huber loss** (smoother than MSE for large errors).
5. A separate **target network** provides stable Q-value targets. It is hard-copied from the online network every 200 update steps.
6. Epsilon decays from 1.0 → 0.05 over training (`epsilon_decay = 0.9995`).

Network architecture: `input(15) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(2)`

Key hyperparameters (`config.yaml` under `dqn:`):
- `lr: 0.001` — Adam learning rate
- `gamma: 0.99` — discount factor (how much future rewards matter)
- `buffer_size: 50000` — replay buffer capacity
- `target_update: 200` — steps between target network syncs

### PPO (`agents/ppo_agent.py`)

**Proximal Policy Optimization** — an on-policy, actor-critic method.

How it works:
1. Collect a **rollout** of 2048 steps using the current policy.
2. Compute **GAE (Generalized Advantage Estimation)** — a credit assignment method that balances bias vs. variance when estimating how good each action was.
3. Normalize advantages across the rollout batch.
4. Run **10 epochs** of mini-batch updates over the collected data. Each update uses the **clipped surrogate objective** to prevent the policy from changing too drastically in one step (clip_epsilon = 0.2).
5. The loss combines: policy gradient + value function error + entropy bonus (encourages exploration).

Network architecture: shared trunk `input(15) → Linear(256) → ReLU → Linear(256) → ReLU`, then two heads — `actor: Linear(256) → Linear(2)` and `critic: Linear(256) → Linear(1)`.

Key hyperparameters (`config.yaml` under `ppo:`):
- `lr: 3e-4` — Adam learning rate
- `gae_lambda: 0.95` — GAE smoothing (higher = lower variance, more bias)
- `clip_epsilon: 0.2` — maximum policy change per update
- `entropy_coef: 0.01` — exploration bonus weight
- `rollout_steps: 2048` — steps collected before each policy update

### Why both DQN and PPO?

DQN and PPO represent two fundamentally different RL paradigms:
- **DQN** learns a value function (how good is each action?) and is sample-efficient due to experience replay.
- **PPO** learns a policy directly and tends to be more stable but less sample-efficient.

Comparing them on the same task demonstrates which paradigm performs better for traffic control and provides a stronger portfolio result.

---

## The Baseline (`baselines/fixed_time.py`)

A **fixed-time controller** that cycles through phases on a 60-second schedule: 30 seconds NS green, 30 seconds EW green, regardless of traffic conditions.

This is what most real-world traffic signals used before adaptive control. It exists purely as a comparison target — if RL cannot beat this, the approach has failed.

---

## Training Loop (`training/train.py`)

Both algorithms share the same entry point. The training loop:
1. Seeds all random number generators for reproducibility (`seed: 42`).
2. Builds the environment and discovers its state/action sizes.
3. Creates the agent and tracking writers (MLflow + TensorBoard).
4. Runs episodes, logging `episode_reward`, `mean_waiting_time`, `mean_queue_length`, and agent-specific metrics (epsilon for DQN, policy/value loss for PPO) at every episode.
5. Saves the **best checkpoint** (highest cumulative reward) and periodic checkpoints every 100 episodes.

To watch training in the SUMO GUI:
```bash
python training/train.py --algo dqn --gui
```

To view metrics during or after training:
```bash
# TensorBoard
tensorboard --logdir results/tensorboard

# MLflow
mlflow ui --backend-store-uri mlruns
# then open http://localhost:5000
```

---

## Evaluation (`training/evaluate.py`)

Runs each controller (Fixed-time, DQN, PPO) for 5 episodes without exploration and computes mean ± std for:
- Mean waiting time (seconds)
- Mean queue length (vehicles)
- Max waiting time (seconds)
- Episode reward

Outputs:
- `results/metrics/evaluation_results.csv`
- `results/plots/comparison.png`

```bash
# Evaluate on a different scenario
python training/evaluate.py --scenario high --episodes 10
```

---

## Experiments

### Experiment 1: Reward Function Ablation (`experiments/reward_ablation.py`)

Trains DQN three times — once with each reward type (`waiting_time`, `queue_length`, `combined`) — and evaluates all three. This answers: **does the choice of reward signal matter, and which performs best in terms of actual waiting time?**

Output: `results/metrics/reward_ablation.csv`, `results/plots/reward_ablation.png`

### Experiment 2: State Feature Ablation (`experiments/state_ablation.py`)

Trains DQN five times with different subsets of state features:

| Variant | Features included |
|---------|-----------------|
| `full` | queue + density + waiting_time + phase (baseline) |
| `no_density` | queue + waiting_time + phase |
| `no_waiting` | queue + density + phase |
| `no_phase` | queue + density + waiting_time |
| `queue_only` | queue + phase only |

This answers: **which features are actually necessary? Can we get away with a simpler state?**

Output: `results/metrics/state_ablation.csv`, `results/plots/state_ablation.png`

---

## Configuration Reference (`configs/config.yaml`)

All behavior is controlled by a single YAML file. Key sections:

| Section | Key | What it does |
|---------|-----|-------------|
| `sumo` | `gui: false` | Set to `true` to watch simulation in SUMO GUI |
| `sumo` | `yellow_time: 3` | Duration of yellow light between phase changes |
| `sumo` | `simulation_seconds: 3600` | How long each simulated episode is |
| `environment` | `delta_time: 5` | Seconds between agent decisions |
| `environment` | `min_green: 10` | Minimum green time before the agent can switch |
| `state` | `use_density: true` | Toggle per-feature inclusion (used by state ablation) |
| `reward` | `type: "waiting_time"` | Reward function: `waiting_time / queue_length / combined / pressure` |
| `dqn` | `epsilon_decay: 0.9995` | Controls how fast exploration decays |
| `training` | `num_episodes: 500` | Total training episodes |
| `training` | `device: "auto"` | `"auto"` uses GPU if available, else CPU |

---

## Outputs Summary

| File | Created by | Contents |
|------|-----------|---------|
| `results/checkpoints/dqn_best.pt` | `train.py --algo dqn` | Best DQN weights |
| `results/checkpoints/ppo_best.pt` | `train.py --algo ppo` | Best PPO weights |
| `results/metrics/evaluation_results.csv` | `evaluate.py` | DQN vs PPO vs baseline metrics |
| `results/metrics/reward_ablation.csv` | `reward_ablation.py` | Per-reward-type metrics |
| `results/metrics/state_ablation.csv` | `state_ablation.py` | Per-feature-set metrics |
| `results/plots/comparison.png` | `evaluate.py` | Bar chart: 3-controller comparison |
| `results/plots/reward_ablation.png` | `reward_ablation.py` | Bar chart: reward variants |
| `results/plots/state_ablation.png` | `state_ablation.py` | Bar chart: state variants |

---

## Adding New Route Scenarios (Future Work)

To experiment with custom traffic patterns, add a new entry to the `scenarios` dict in `network/generate_network.py`:

```python
"my_scenario": {
    "N2S": 400, "S2N": 200,   # vehicles per hour, N→S and S→N through traffic
    "E2W": 400, "W2E": 400,
    "N2E": 80,  "N2W": 80,    # right/left turns
    "S2E": 80,  "S2W": 80,
    "E2N": 80,  "E2S": 80,
    "W2N": 80,  "W2S": 80,
},
```

Then re-run `python network/generate_network.py`. The file `network/single/routes_my_scenario.rou.xml` will be created. You can then pass `--scenario my_scenario` to `train.py` and `evaluate.py`.

---

## Phase 6 (Future): Multi-Agent 3×3 Grid

The network generator has a `--grid` flag stub ready. Phase 6 will extend this to a 3×3 grid of 9 coordinated intersections, requiring one agent per intersection and a shared observation space design.
