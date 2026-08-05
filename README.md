# Deep RL Traffic Signal Control (DQN & PPO)

A reinforcement learning system that learns to control a traffic signal at a 4-way intersection using [SUMO](https://sumo.dlr.de/) (Simulation of Urban MObility) and PyTorch. Two RL algorithms — **DQN** and **PPO** — are implemented from scratch and benchmarked against a fixed-time baseline across multiple traffic scenarios, with reward-function and state-feature ablation studies.

## Results at a Glance

Both RL agents cut mean vehicle waiting time by **~70–73%** versus a classic fixed-time (30s/30s) signal controller, evaluated over 5 held-out traffic realizations (mean ± std):

| Controller | Mean wait (s) | vs. baseline |
|---|:---:|:---:|
| Fixed-time (baseline) | 49.1 ± 1.0 | — |
| DQN | 14.5 ± 1.1 | **−70.4%** |
| PPO | 13.2 ± 0.8 | **−73.1%** |

![Controller comparison](results/plots/comparison.png)

**Training dynamics differ sharply between the two algorithms** — PPO converges to near-optimal performance by ~episode 50, while DQN takes ~200 episodes to catch up, consistent with PPO's on-policy updates giving smoother (if less sample-efficient) learning versus DQN's epsilon-greedy exploration:

![Training curves](results/plots/training_curves.png)

Evaluation runs each controller on the **same set of held-out traffic seeds** (Poisson vehicle arrivals — every episode is a different traffic realization, disjoint from the seeds used in training), so the error bars reflect genuine variability across traffic conditions rather than a replayed identical episode.

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
│   ├── evaluate.py              # Runs trained agents, compares to baseline, saves plots
│   └── plot_curves.py           # Training-curve plots from MLflow run history
├── tests/
│   ├── test_fixed_time.py       # Baseline schedule correctness
│   ├── test_dqn.py              # Q-network, replay buffer, epsilon schedule
│   └── test_ppo.py              # Actor-critic shapes, GAE correctness + bootstrapping
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

# 2. Run the unit tests (no SUMO needed)
python -m pytest tests/

# 3. Generate the SUMO network and route files
python network/generate_network.py

# 4. Train DQN (500 episodes, ~1 hr)
python training/train.py --algo dqn

# 5. Train PPO (500 episodes, ~45 min)
python training/train.py --algo ppo

# 6. Evaluate both agents vs. fixed-time baseline
python training/evaluate.py

# 7. Plot training curves from the MLflow logs
python training/plot_curves.py

# 8. Reward function ablation experiment
python experiments/reward_ablation.py

# 9. State feature ablation experiment
python experiments/state_ablation.py
```

All steps are idempotent — re-running overwrites previous outputs.

---

## The Simulation

**SUMO** simulates a single 4-way intersection with a traffic light named `center`. There are 4 incoming edges (N2C, S2C, E2C, W2C) and 4 outgoing edges. Each edge has 2 lanes. Speed limit is 50 km/h (13.89 m/s). Each edge is 300 m long.

Each simulated episode covers **1 hour of traffic** (3600 simulated seconds). The RL agent makes a decision every 5 seconds (`delta_time = 5`), giving 720 steps per episode.

**Episodes are stochastic.** Vehicle arrivals are Poisson processes (`period="exp(rate)"` in the route files) and drivers have randomized speed factors, so each episode — driven by a per-episode SUMO seed — is a different traffic realization. Training uses seeds `42 + episode`; evaluation uses a disjoint held-out seed range (`10000+`), the same for every controller so comparisons are paired.

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

**Minimum green enforcement** — the agent cannot switch phases until the current phase has accumulated at least `min_green` (10) seconds of green. If it tries, the action is overridden to keep the current phase. Because decisions happen on a 5-second grid and a switch step yields only 2 s of green (3 s go to yellow), green time quantizes to 2+5k seconds — so the effective minimum is 12 s, the smallest reachable value ≥ 10.

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
6. Epsilon decays from 1.0 → 0.05 **per episode** (`epsilon_decay = 0.99`, floor reached ~episode 300 of 500). Decaying per learn-step — the common tutorial pattern — would exhaust exploration within ~8 episodes here (720 learn steps/episode).
7. Episodes end by **time limit**, which is not a true terminal state — so the Q-target still bootstraps `max Q(s′,·)` across episode boundaries instead of zeroing it (Pardo et al. 2018, *Time Limits in Reinforcement Learning*).

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
2. Compute **GAE (Generalized Advantage Estimation)** — a credit assignment method that balances bias vs. variance when estimating how good each action was. When the rollout cuts off mid-episode, the tail is **bootstrapped with the critic's `V(s)`** rather than treated as a terminal; time-limit truncations inside a rollout are handled with partial-episode bootstrapping (the discounted `V(s_next)` is folded into the final reward).
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

> **Fair comparison:** every controller is evaluated on the *same* held-out traffic seeds (10000–10004), disjoint from the training seed range — so differences between controllers are paired comparisons across identical traffic demand, and the reported std reflects real episode-to-episode variability.

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

![Reward ablation](results/plots/reward_ablation.png)

| Reward type | Mean wait (s) |
|---|:---:|
| `waiting_time` | 14.3 ± 1.2 |
| `queue_length` | 15.0 ± 0.9 |
| `combined` | 13.4 ± 1.1 |

`combined` is nominally best but within one std of `waiting_time`; `queue_length` alone is the weakest signal, plausibly because it ignores *how long* vehicles have been waiting, only how many are currently stopped. All three comfortably beat the fixed-time baseline (49.1s), so reward choice matters far less than the decision to use RL at all here.

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

![State ablation](results/plots/state_ablation.png)

| Variant | Mean wait (s) |
|---|:---:|
| `full` | 14.3 ± 1.2 |
| `no_density` | 13.5 ± 0.8 |
| `no_waiting` | 12.5 ± 0.5 |
| `no_phase` | 13.1 ± 1.1 |
| `queue_only` | 13.2 ± 1.0 |

Key finding: **on this task, none of the five state variants beat the others outside of noise** — every variant's mean ± std overlaps with every other's. Even `queue_only` (queue length + phase, the simplest state tested) performs indistinguishably from `full`. This is itself informative: it means queue length is close to a sufficient statistic for this single-intersection MDP, and the extra features (density, cumulative waiting time) aren't earning their keep here. (An earlier version of this ablation, run before evaluation was made stochastic, showed a spurious ranking — see [Correctness Notes](#correctness-notes).) A harder setting — the multi-intersection grid, or a state-aliasing scenario designed to need `phase` — would be needed to actually separate these variants.

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
| `dqn` | `epsilon_decay: 0.99` | Per-episode exploration decay (floor ~episode 300) |
| `training` | `num_episodes: 500` | Total training episodes |
| `training` | `log_interval: 25` | Print progress every N episodes |
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

## Correctness Notes

RL implementations fail silently — the agent still "learns something" even when the math is subtly wrong. Three such issues were found and fixed in this project; they're documented here because they're instructive:

1. **Time-limit truncation ≠ terminal state.** Episodes end after 1 simulated hour, but traffic doesn't cease to exist at that moment. Treating the cutoff as a terminal state (`done=1` in the Bellman target) teaches the agent that the world ends at step 720, biasing values near episode end. Fix: DQN bootstraps `max Q(s′,·)` across truncations; PPO folds `γ·V(s_next)` into the final reward (partial-episode bootstrapping, Pardo et al. 2018).
2. **GAE must bootstrap the rollout tail.** PPO's 2048-step rollouts cut off mid-episode. The last transition's advantage must use the critic's `V(s)` for the state the env is left in — hard-coding 0 there treats every rollout boundary as an episode end.
3. **Per-step epsilon decay exhausts exploration almost immediately.** With 720 learn steps per episode, a 0.9995 per-step decay hits the exploration floor after ~8 of 500 episodes. Decay is per-episode (0.99), reaching the floor around episode 300.

Additionally, evaluation originally used deterministic, equally-spaced vehicle insertions — every "episode" was the identical scenario replayed, making mean ± std over 5 episodes meaningless (std was exactly 0). Route files now use Poisson arrivals with per-episode SUMO seeds, so evaluation statistics reflect genuine variability. One side effect worth calling out: with stochastic evaluation, the state-ablation ranking from before this fix no longer holds — see the [State Feature Ablation](#experiment-2-state-feature-ablation-experimentsstate_ablationpy) results below, which now show the five variants performing indistinguishably rather than a clear "phase matters most" trend. That original conclusion was almost certainly noise from a single deterministic rollout.

**A known failure mode, found via the training curves, not hidden from them:** DQN episode 448 (seed 490) shows a queue length of 16.3 vehicles and mean waiting time of 1275s — both roughly 5–100x the typical value. SUMO's `time-to-teleport` is disabled (`-1`) in this project so gridlocked vehicles are never force-removed, which is realistic but means a sufficiently bad sequence of actions — most likely a random exploration action landing at the wrong moment during a high-arrival-rate seed — can trigger a queue buildup the agent doesn't clear before the episode's 1-hour time limit. It's a single episode out of 500 and training recovers immediately after, but it's a real limitation of a fixed-horizon single-intersection formulation: there's no mechanism forcing the agent to prioritize clearing an already-large queue over marginal further reward. `training/plot_curves.py` clips the y-axis to the 1st–99th percentile so this doesn't dominate the plot, and annotates which point was clipped.

## Testing

```bash
python -m pytest tests/
```

18 unit tests cover the pieces that can fail silently: fixed-time schedule arithmetic, replay-buffer capacity/shapes, epsilon schedule (including a regression test for the per-step-decay bug), Q-network/actor-critic output shapes, and hand-verified GAE computations (reward-to-go, episode-boundary cutting, tail bootstrapping). No SUMO installation is needed to run them.

## Roadmap: Multi-Agent 3×3 Grid

The network generator has a `--grid` flag stub ready. The next phase extends this to a 3×3 grid of 9 coordinated intersections — one agent per intersection with a shared observation-space design, which is where the DQN-vs-PPO comparison should genuinely diverge.

## License

MIT — see [LICENSE](LICENSE).
