# Technical Reference

Full details behind the [top-level README](../README.md): the simulation, the
environment spec, both agents, both baselines, the training/evaluation loops,
configuration reference, correctness notes, and the roadmap. If you just want
the results and the story, see the [case study](https://claude.ai/code/artifact/1450e324-cde9-4644-a414-7607c999f9f5)
instead — this page is the reference you reach for when you're extending the
code or running it yourself.

---

## The Simulation

**SUMO** simulates a single 4-way intersection with a traffic light named `center`.
There are 4 incoming edges (N2C, S2C, E2C, W2C) and 4 outgoing edges. Each incoming
edge has 3 lanes (dedicated left-turn lane, through, through+right); outgoing edges
have 2. Speed limit is 50 km/h (13.89 m/s). Each edge is 300 m long.

Each simulated episode covers **1 hour of traffic** (3600 simulated seconds). The
RL agent makes a decision every 5 seconds (`delta_time = 5`), giving 720 steps per
episode.

**Episodes are stochastic.** Vehicle arrivals are Poisson processes
(`period="exp(rate)"` in the route files) and drivers have randomized speed
factors, so each episode — driven by a per-episode SUMO seed — is a different
traffic realization. Training uses seeds `42 + episode`; evaluation uses a
disjoint held-out seed range (`10000+`), the same for every controller so
comparisons are paired.

## Network: Protected Left Turns

The original network gave each approach 2 shared lanes and the signal only 2
phases (NS / EW), so left-turning traffic just merged into the same phase as
through traffic. Each incoming approach now has **3 lanes** (dedicated left-turn
lane, through lane, through+right lane) and **2 lanes outgoing**. SUMO's default
lane-connection guesser assigns lane-to-movement connections 1:1 when lane count
matches destination count, and its default TLS builder responds to the dedicated
left lane by adding **protected left-turn phases** automatically — no explicit
connection/phase-logic file needed. This produces **4 green phases** (NS main
including permitted left, NS protected-left, EW main, EW protected-left) instead
of 2. `traffic_env.py`'s `_discover_phases()` reads whatever phases came out of
the network, so the action space (2 → 4) and state size (15 → 17, the phase
one-hot grew) adapt automatically — no agent code changes were needed, only
`network/generate_network.py`.

One consequence worth flagging: more phases means more ways for training to go
briefly wrong. A near-random policy can leave a direction starved of green long
enough, combined with SUMO's `time-to-teleport` being disabled (see
[Correctness Notes](#correctness-notes)), to gridlock a queue that doesn't clear
for the rest of the episode. This isn't a bug — a reasonable fixed-time schedule
keeps the same network well under control — but training stability over the first
~100-200 episodes is worth watching, not just the final result.

---

## The Environment (`env/traffic_env.py`)

The environment follows the [Gymnasium](https://gymnasium.farama.org/) API
(`reset`, `step`, `close`).

### Observation (State)

At each step, the agent sees a vector built from the following features, all
normalized to [0, 1]:

| Feature | Source | Normalization cap |
|---------|--------|-----------------|
| Queue length per edge (×4) | Halting vehicles on each incoming edge | 30 vehicles |
| Density per edge (×4) | Lane occupancy percentage | 100% |
| Waiting time per edge (×4) | Cumulative seconds vehicles have been waiting | 300 s |
| Current phase (one-hot, ×4) | Which green phase is active | — |
| Phase duration (×1) | Seconds in current phase | 120 s |

Default state vector size: **17 values** (4+4+4+4+1). The phase one-hot width —
and therefore total state size and action-space size — is auto-detected from
whatever the generated network's traffic light program actually contains
(`_discover_phases()` in `traffic_env.py`), not hardcoded.

### Action

`Discrete(4)` — choose which green phase to activate. Auto-discovered from the
network's traffic light program, not hardcoded:
- Phase 0: NS main (through + right, left permitted/yielding)
- Phase 1: NS protected left turn
- Phase 2: EW main (through + right, left permitted/yielding)
- Phase 3: EW protected left turn

**Yellow transitions are inserted automatically** — when the agent switches
phases, the environment inserts a 3-second yellow before activating the new
green. The agent does not need to handle this.

**Minimum green enforcement** — the agent cannot switch phases until the current
phase has accumulated at least `min_green` (10) seconds of green. If it tries,
the action is overridden to keep the current phase. Because decisions happen on
a 5-second grid and a switch step yields only 2 s of green (3 s go to yellow),
green time quantizes to 2+5k seconds — so the effective minimum is 12 s, the
smallest reachable value ≥ 10.

### Reward

Five options, set via `reward.type` in `config.yaml`:

| Type | Formula | What it optimizes |
|------|---------|-----------------|
| `waiting_time` | `−Σ waiting_time` across all edges | Minimize total seconds vehicles wait |
| `queue_length` | `−Σ queue_length` across all edges | Minimize total vehicles stuck |
| `combined` | `−(0.5 × waiting + 0.5 × queue)` | Weighted combination |
| `pressure` | `−|queue_red_edges − queue_green_edges|` | Serve the most congested direction |
| `delta_waiting` | `waiting_time[t-1] − waiting_time[t]` | Change in cumulative delay (sumo-rl's default reward) |

`delta_waiting` is bounded per-step regardless of how congested the intersection
has become, unlike `waiting_time`/`queue_length`/`combined` which grow with the
absolute backlog. See the [reward ablation](#experiment-1-reward-function-ablation)
below for why that turns out to matter.

All rewards are scaled by `reward.scale = 0.001` to keep values in a stable range
for neural network training.

---

## Traffic Scenarios (Route Files)

Four scenarios are generated by `generate_network.py`:

| File | Description | N↕S flow | E↔W flow |
|------|------------|---------|---------|
| `routes_normal.rou.xml` | Balanced traffic, moderate volume | 300 veh/hr | 300 veh/hr |
| `routes_ns_peak.rou.xml` | Morning rush, North-South dominant | 600 veh/hr | 200 veh/hr |
| `routes_ew_peak.rou.xml` | Evening rush, East-West dominant | 200 veh/hr | 600 veh/hr |
| `routes_high.rou.xml` | High congestion, all directions saturated | 800 veh/hr | 800 veh/hr |

```bash
python training/train.py --algo dqn --scenario ns_peak
python training/evaluate.py --scenario ew_peak
```

**To add a new scenario:** open `network/generate_network.py`, add an entry to
the `scenarios` dict in `write_route_file()`:

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

Then re-run `python network/generate_network.py`. The new `.rou.xml` file will
appear in `network/single/`, and `--scenario my_scenario` works on `train.py`
and `evaluate.py`. No other files need to change.

---

## The Agents

### DQN (`agents/dqn_agent.py`)

**Deep Q-Network** — an off-policy, value-based method.

1. At each step, pick an action using **epsilon-greedy**: with probability ε,
   explore randomly; otherwise, pick the action with the highest predicted
   Q-value.
2. Store `(state, action, reward, next_state, done)` in a **replay buffer**
   (capacity 50,000).
3. Every step, sample a random mini-batch of 64 experiences and update the
   Q-network via the Bellman equation: `Q(s,a) = r + γ · max Q'(s', a')`.
4. Loss is **Huber loss** (smoother than MSE for large errors).
5. A separate **target network** provides stable Q-value targets, hard-copied
   from the online network every 200 update steps.
6. Epsilon decays from 1.0 → 0.05 **per episode** (`epsilon_decay = 0.99`, floor
   reached ~episode 300 of 500) — decaying per learn-step would exhaust
   exploration within ~8 episodes here (720 learn steps/episode).
7. Episodes end by **time limit**, not a true terminal state, so the Q-target
   still bootstraps `max Q(s′,·)` across episode boundaries instead of zeroing
   it (Pardo et al. 2018, *Time Limits in Reinforcement Learning*).

Network architecture: `input(17) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(4)`

Key hyperparameters (`config.yaml` under `dqn:`):
- `lr: 0.001` — Adam learning rate
- `gamma: 0.99` — discount factor
- `buffer_size: 50000` — replay buffer capacity
- `target_update: 200` — steps between target network syncs

### PPO (`agents/ppo_agent.py`)

**Proximal Policy Optimization** — an on-policy, actor-critic method.

1. Collect a **rollout** of 2048 steps using the current policy.
2. Compute **GAE (Generalized Advantage Estimation)**. When the rollout cuts off
   mid-episode, the tail is **bootstrapped with the critic's `V(s)`** rather than
   treated as a terminal.
3. Normalize advantages across the rollout batch.
4. Run **10 epochs** of mini-batch updates using the **clipped surrogate
   objective** (`clip_epsilon = 0.2`).
5. Loss combines: policy gradient + value function error + entropy bonus.

Network architecture: shared trunk `input(17) → Linear(256) → ReLU → Linear(256) → ReLU`,
then two heads — `actor: Linear(256) → Linear(4)` and `critic: Linear(256) → Linear(1)`.

Key hyperparameters (`config.yaml` under `ppo:`):
- `lr: 3e-4` — Adam learning rate
- `gae_lambda: 0.95` — GAE smoothing
- `clip_epsilon: 0.2` — max policy change per update
- `entropy_coef: 0.01` — exploration bonus weight
- `rollout_steps: 2048` — steps collected before each policy update

### Why both?

DQN learns a value function and is sample-efficient via experience replay; PPO
learns a policy directly and tends to be more stable but less sample-efficient.
Comparing them on the same task shows which paradigm wins for traffic control
rather than assuming one.

---

## The Baselines

### Fixed-time (`baselines/fixed_time.py`)

Cycles through phases with an equal time split — 15 seconds each in a 60-second
cycle across 4 phases — regardless of traffic conditions. Deliberately naive:
with more phases, an equal split gets *more* wasteful (protected-left phases
don't need as much green time as the main through phases).

### Max-Pressure (`baselines/max_pressure.py`)

A **static, non-learned** controller — at every decision point, it picks the
candidate phase whose served movements have the largest "pressure": queued
vehicles on the incoming lanes that phase would serve, minus queued vehicles on
the corresponding outgoing lanes. Max-Pressure ([Varaiya, 2013](https://en.wikipedia.org/wiki/Max-pressure_controller))
requires no training and is provably throughput-maximizing under mild
assumptions; it's the standard strong baseline used throughout the
traffic-signal-RL literature (RESCO, PressLight, MPLight).

The pressure computation is factored into a pure function
(`compute_phase_pressures`) so it's unit-testable without a running SUMO
instance — see `tests/test_max_pressure.py`.

---

## Training Loop (`training/train.py`)

1. Seeds all RNGs for reproducibility (`seed: 42`).
2. Builds the environment and discovers its state/action sizes.
3. Creates the agent and tracking writers (MLflow + TensorBoard).
4. Runs episodes, logging `episode_reward`, `mean_waiting_time`,
   `mean_queue_length`, and agent-specific metrics at every episode.
5. Saves the **best checkpoint** (highest cumulative reward) and periodic
   checkpoints every 100 episodes.

```bash
python training/train.py --algo dqn --gui       # watch in SUMO GUI
tensorboard --logdir results/tensorboard          # live metrics
mlflow ui --backend-store-uri mlruns               # then open localhost:5000
```

## Evaluation (`training/evaluate.py`)

Runs each controller (Fixed-time, Max-Pressure, DQN, PPO) for 5 episodes without
exploration on the *same* held-out traffic seeds (10000–10004), disjoint from
training, and computes mean ± std for mean waiting time, mean queue length, max
waiting time, and episode reward.

```bash
python training/evaluate.py --scenario high --episodes 10
python training/evaluate.py --scenarios normal,high   # produces comparison_high.png too
```

---

## Experiments

### Experiment 1: Reward Function Ablation (`experiments/reward_ablation.py`)

Trains DQN four times — once per reward type (`waiting_time`, `queue_length`,
`combined`, `delta_waiting`) — and evaluates all four.

![Reward ablation](../results/plots/reward_ablation.png)

| Reward type | Mean wait (s) |
|---|:---:|
| `waiting_time` | 12.2 ± 0.9 |
| `queue_length` | 11.7 ± 0.9 |
| `combined` | 13.3 ± 1.2 |
| `delta_waiting` | 277,377 ± 6,502 (**total gridlock**) |

The first three are statistically indistinguishable. `delta_waiting` trains
smoothly (bounded per-step reward, loss curves look normal) but the resulting
*greedy* policy gridlocks the intersection at evaluation time — because
`delta_waiting = waiting[t-1] − waiting[t]` only rewards the marginal change in
congestion, a policy that keeps the intersection permanently near-saturated can
still score close to zero every step. Worth remembering if citing this project:
`delta_waiting` needs to be paired with something that anchors absolute
congestion (e.g. `combined` with a delta component) to avoid this failure mode.

### Experiment 2: State Feature Ablation (`experiments/state_ablation.py`)

Trains DQN five times with different feature subsets:

| Variant | Features included |
|---------|-----------------|
| `full` | queue + density + waiting_time + phase (baseline) |
| `no_density` | queue + waiting_time + phase |
| `no_waiting` | queue + density + phase |
| `no_phase` | queue + density + waiting_time |
| `queue_only` | queue + phase only |

![State ablation](../results/plots/state_ablation.png)

| Variant | Mean wait (s) |
|---|:---:|
| `full` | 12.5 ± 0.9 |
| `no_density` | 12.2 ± 1.1 |
| `no_waiting` | 12.4 ± 0.6 |
| `no_phase` | 13.7 ± 0.7 |
| `queue_only` | 12.1 ± 1.2 |

Four of five variants are statistically indistinguishable; `no_phase` is the
exception (nominally worse, visibly larger max-wait: 165s vs. ~90-100s in the
underlying CSV) — consistent with not knowing which direction has right-of-way
being the one thing a controller genuinely can't do without. `queue_only`
performing on par with `full` means queue length is close to a sufficient
statistic for this single-intersection MDP.

---

## Configuration Reference (`configs/config.yaml`)

| Section | Key | What it does |
|---------|-----|-------------|
| `sumo` | `gui: false` | Set to `true` to watch simulation in SUMO GUI |
| `sumo` | `yellow_time: 3` | Duration of yellow light between phase changes |
| `sumo` | `simulation_seconds: 3600` | How long each simulated episode is |
| `environment` | `delta_time: 5` | Seconds between agent decisions |
| `environment` | `min_green: 10` | Minimum green time before the agent can switch |
| `state` | `use_density: true` | Toggle per-feature inclusion (used by state ablation) |
| `reward` | `type: "waiting_time"` | Reward function |
| `dqn` | `epsilon_decay: 0.99` | Per-episode exploration decay (floor ~episode 300) |
| `training` | `num_episodes: 500` | Total training episodes |
| `training` | `log_interval: 25` | Print progress every N episodes |
| `training` | `device: "auto"` | `"auto"` uses GPU if available, else CPU |

## Outputs Summary

| File | Created by | Contents |
|------|-----------|---------|
| `results/checkpoints/dqn_best.pt` | `train.py --algo dqn` | Best DQN weights |
| `results/checkpoints/ppo_best.pt` | `train.py --algo ppo` | Best PPO weights |
| `results/checkpoints/{algo}_latest.pt`, `{algo}_progress.json` | `train.py --resume` | Resume checkpoint + episode/best-reward sidecar |
| `results/metrics/evaluation_results.csv` | `evaluate.py` | Fixed-time vs Max-Pressure vs DQN vs PPO, `normal` scenario |
| `results/metrics/evaluation_results_high.csv` | `evaluate.py --scenarios normal,high` | Same, `high`-demand scenario |
| `results/metrics/reward_ablation.csv` | `reward_ablation.py` | Per-reward-type metrics |
| `results/metrics/state_ablation.csv` | `state_ablation.py` | Per-feature-set metrics |
| `results/plots/comparison.png` / `comparison_high.png` | `evaluate.py` | Bar chart: 4-controller comparison, per scenario |
| `results/plots/training_curves.png` | `plot_curves.py` | Episode reward / waiting time vs. episode, DQN vs PPO |
| `results/plots/reward_ablation.png` | `reward_ablation.py` | Bar chart: reward variants |
| `results/plots/state_ablation.png` | `state_ablation.py` | Bar chart: state variants |

---

## Correctness Notes

RL implementations fail silently — the agent still "learns something" even when
the math is subtly wrong. Three such issues were found and fixed; they're
documented here because they're instructive:

1. **Time-limit truncation ≠ terminal state.** Episodes end after 1 simulated
   hour, but traffic doesn't cease to exist at that moment. Treating the cutoff
   as a terminal state (`done=1` in the Bellman target) teaches the agent that
   the world ends at step 720, biasing values near episode end. Fix: DQN
   bootstraps `max Q(s′,·)` across truncations; PPO folds `γ·V(s_next)` into the
   final reward (partial-episode bootstrapping, Pardo et al. 2018).
2. **GAE must bootstrap the rollout tail.** PPO's 2048-step rollouts cut off
   mid-episode. The last transition's advantage must use the critic's `V(s)` for
   the state the env is left in — hard-coding 0 there treats every rollout
   boundary as an episode end.
3. **Per-step epsilon decay exhausts exploration almost immediately.** With 720
   learn steps per episode, a 0.9995 per-step decay hits the exploration floor
   after ~8 of 500 episodes. Decay is per-episode (0.99), reaching the floor
   around episode 300.

Additionally, evaluation originally used deterministic, equally-spaced vehicle
insertions — every "episode" was the identical scenario replayed, making mean ±
std over 5 episodes meaningless (std was exactly 0). Route files now use Poisson
arrivals with per-episode SUMO seeds. One side effect: with stochastic
evaluation, the state-ablation ranking from before this fix no longer holds —
the five variants now perform indistinguishably rather than showing a clear
"phase matters most" trend, which was almost certainly noise from a single
deterministic rollout.

**A known failure mode, found via the training curves, not hidden from them:**
DQN episode 448 (seed 490) shows a queue length of 16.3 vehicles and mean
waiting time of 1275s — both roughly 5–100x the typical value. SUMO's
`time-to-teleport` is disabled (`-1`) in this project so gridlocked vehicles are
never force-removed, which is realistic but means a sufficiently bad sequence of
actions can trigger a queue buildup the agent doesn't clear before the episode's
1-hour time limit. It's a single episode out of 500 and training recovers
immediately after, but it's a real limitation of a fixed-horizon
single-intersection formulation. `training/plot_curves.py` clips the y-axis to
the 1st–99th percentile so this doesn't dominate the plot, and annotates which
point was clipped.

## Resumable Training (`training/train.py --resume`, `scripts/run_pipeline.py`)

A full training run (500 episodes) takes over an hour — long enough to outlast
whatever environment it's running in. Every `latest_interval` episodes (config:
`training.latest_interval`, default 20), the training loop saves
`{algo}_latest.pt` plus a small JSON progress sidecar. `--resume` picks both back
up: an interrupted run loses at most `latest_interval` episodes rather than
restarting from scratch. `experiments/reward_ablation.py --resume` and
`experiments/state_ablation.py --resume` get the same behavior for free.

`scripts/run_pipeline.py` chains network generation → DQN training → PPO
training → evaluation → training-curve plotting → reward ablation → state
ablation, always passing `--resume`, and supports `--start-from "<step name>"`
to skip already-completed steps after an interruption:

```bash
python scripts/run_pipeline.py                          # full run, ~5-6 hrs
python scripts/run_pipeline.py --core-only               # skip ablations, ~2 hrs
python scripts/run_pipeline.py --start-from "Train PPO"  # resume after an interruption
```

Two infrastructure bugs surfaced (and got fixed) while stress-testing this
against repeated interruptions:

- **Matplotlib's default GUI backend can hang in a headless/background
  process.** Every script that plots now calls `matplotlib.use("Agg")` before
  importing `pyplot`.
- **Streaming a subprocess's output without an explicit encoding silently uses
  the wrong one.** `subprocess.Popen(..., text=True)` without `encoding=` decodes
  the child's stdout using the *parent's* default locale (`cp1252` on Windows),
  not whatever the child actually writes (`utf-8`) — non-ASCII output crashed the
  parent outright. Fixed by passing `encoding="utf-8", errors="replace"`
  explicitly.

## Roadmap: Multi-Agent 3×3 Grid

The network generator has a `--grid` flag stub ready. The next phase extends
this to a 3×3 grid of 9 coordinated intersections — one agent per intersection
with a shared observation-space design, which is where the DQN-vs-PPO comparison
should genuinely diverge.
