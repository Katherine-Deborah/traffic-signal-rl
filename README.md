# Deep RL Traffic Signal Control (DQN & PPO)

[![tests](https://github.com/Katherine-Deborah/traffic-signal-rl/actions/workflows/tests.yml/badge.svg)](https://github.com/Katherine-Deborah/traffic-signal-rl/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-informational)

DQN and PPO, implemented from scratch in PyTorch, learn to run a 4-way traffic
signal in [SUMO](https://sumo.dlr.de/) — cutting average wait time 91% versus a
naive fixed-time schedule, and roughly matching **Max-Pressure**, the strong
non-learned baseline used throughout the traffic-signal-RL literature.

**[→ Read the case study](https://claude.ai/code/artifact/1450e324-cde9-4644-a414-7607c999f9f5)**
for the full story — the intersection mechanics, the correctness bugs found
along the way, and the honest failure case under saturated traffic.

## Results at a glance

Evaluated on 5 held-out traffic seeds (mean ± std seconds of wait per vehicle):

| Controller | Mean wait (s) | vs. fixed-time |
|---|:---:|:---:|
| Fixed-time (naive) | 134.9 ± 4.0 | — |
| Max-Pressure (strong baseline) | 12.4 ± 1.1 | −90.8% |
| DQN | 11.9 ± 0.9 | −91.2% |
| PPO | 11.8 ± 1.0 | −91.2% |

![Controller comparison](results/plots/comparison.png)

RL ties the strong baseline on ordinary traffic — but loses badly on traffic
heavier than what it trained on. See the [case study](https://claude.ai/code/artifact/1450e324-cde9-4644-a414-7607c999f9f5#catch)
for that result and the reward-shaping failure mode found alongside it.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/                     # 24 unit tests, no SUMO needed
python network/generate_network.py          # generate the SUMO network + routes
python training/train.py --algo dqn         # ~1 hr
python training/train.py --algo ppo         # ~45 min
python training/evaluate.py                 # compare vs. Fixed-time + Max-Pressure
python training/plot_curves.py
```

Or run the whole pipeline (network → both agents → evaluation → ablations),
resumable if interrupted:

```bash
python scripts/run_pipeline.py --core-only   # ~2 hrs, skips ablations
```

## How it works

**Environment** (`env/traffic_env.py`) wraps SUMO in the [Gymnasium](https://gymnasium.farama.org/)
API: every 5 seconds the agent sees a 17-value state (queue length, density,
waiting time, and current phase per approach) and picks one of 4 signal phases —
including two **protected left-turn phases**, made possible by giving each
approach a dedicated left-turn lane. Action space and state size are
auto-discovered from the network, not hardcoded.

**Agents** — DQN (`agents/dqn_agent.py`), an off-policy value-based method with
a replay buffer and target network, and PPO (`agents/ppo_agent.py`), an
on-policy actor-critic method with GAE and a clipped surrogate objective — are
both implemented from scratch and trained independently on the same environment.

**Baselines** — a naive fixed-time schedule and **Max-Pressure**
(`baselines/max_pressure.py`), a static non-learned controller that's the
standard strong comparison point in the traffic-signal-RL literature.

Full technical reference — hyperparameters, reward/state options, config
reference, correctness notes, and the roadmap — lives in
**[docs/DETAILS.md](docs/DETAILS.md)**.

## Project structure

```
traffic/
├── configs/config.yaml       # All hyperparameters and paths
├── env/traffic_env.py        # Gymnasium environment wrapping SUMO via traci
├── agents/                   # DQN and PPO, implemented from scratch
├── baselines/                # Fixed-time and Max-Pressure controllers
├── network/                  # SUMO network/route generation
├── training/                 # train.py, evaluate.py, plot_curves.py
├── experiments/               # Reward and state-feature ablations
├── scripts/run_pipeline.py   # Full pipeline, resumable via --start-from
├── tests/                    # 24 unit tests, no SUMO needed
└── results/                  # Checkpoints, metrics, plots
```

## Testing

```bash
python -m pytest tests/
```

24 unit tests cover the pieces that fail silently: replay-buffer shapes, the
epsilon schedule, hand-verified GAE computations, and Max-Pressure's
phase-pressure arithmetic — all runnable without a SUMO installation.

## License

MIT — see [LICENSE](LICENSE).
