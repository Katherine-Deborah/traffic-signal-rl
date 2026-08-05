"""
Full retraining pipeline — runs every step in order and logs to results/pipeline.log.

Usage
-----
    python scripts/run_pipeline.py                     # everything (~5 hrs)
    python scripts/run_pipeline.py --core-only          # train + eval + curves (~2 hrs)
    python scripts/run_pipeline.py --start-from "Train PPO"   # resume after an interruption

Appends to results/pipeline.log rather than overwriting, so a resumed run keeps
the history of the earlier attempt.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORE_STEPS = [
    ("Generate network", [sys.executable, "network/generate_network.py"]),
    # --resume is always passed: it's a no-op that starts fresh when there's
    # no {algo}_latest.pt/progress.json yet, and picks back up mid-training
    # (losing at most latest_interval episodes) if this step itself gets
    # interrupted and the whole pipeline is relaunched with --start-from.
    ("Train DQN", [sys.executable, "training/train.py", "--algo", "dqn", "--resume"]),
    ("Train PPO", [sys.executable, "training/train.py", "--algo", "ppo", "--resume"]),
    ("Evaluate", [sys.executable, "training/evaluate.py", "--scenarios", "normal,high"]),
    ("Plot training curves", [sys.executable, "training/plot_curves.py"]),
]

ABLATION_STEPS = [
    ("Reward ablation", [sys.executable, "experiments/reward_ablation.py", "--resume"]),
    ("State ablation", [sys.executable, "experiments/state_ablation.py", "--resume"]),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--core-only", action="store_true", help="Skip ablation experiments")
    p.add_argument(
        "--start-from", default=None,
        help="Step name to resume from (skips every step before it). "
             "Must match a step name exactly, e.g. 'Train PPO'.",
    )
    args = p.parse_args()

    steps = CORE_STEPS + ([] if args.core_only else ABLATION_STEPS)

    if args.start_from:
        names = [name for name, _ in steps]
        if args.start_from not in names:
            print(f"Unknown step '{args.start_from}'. Valid: {names}")
            sys.exit(1)
        steps = steps[names.index(args.start_from):]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Piped stdout (not a TTY) makes Python default to block-buffering, so
    # print() calls without flush=True (most of the ones in train.py etc.)
    # would sit invisible in the child's internal buffer for a while even
    # with the streaming above. Force unbuffered so lines appear as soon as
    # they're printed.
    env["PYTHONUNBUFFERED"] = "1"

    log_path = os.path.join(ROOT, "results", "pipeline.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    t_start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        if args.start_from:
            resume_banner = f"\n\n{'#' * 60}\n  RESUMING from '{args.start_from}'\n{'#' * 60}\n"
            print(resume_banner, flush=True)
            log.write(resume_banner)
        for i, (name, cmd) in enumerate(steps, 1):
            banner = f"\n{'=' * 60}\n  [{i}/{len(steps)}] {name}\n{'=' * 60}\n"
            print(banner, flush=True)
            log.write(banner)
            log.flush()

            # Stream the child's output through *this* process's own stdout
            # (in addition to the log file) instead of redirecting it
            # straight to a file descriptor. subprocess.run(stdout=log) sends
            # child output directly to the file, invisible to our own
            # stdout — which means whatever supervises this backgrounded
            # process sees total silence for minutes at a time (e.g. between
            # DQN's every-25-episode progress prints). If anything treats
            # prolonged stdout silence as "hung" and kills it, that silence
            # is the trigger, not the training itself. Tee-ing every line
            # through print(..., flush=True) keeps continuous, real activity
            # visible on our own stdout throughout each step.
            # encoding="utf-8" is required here: with text=True but no
            # explicit encoding, Python decodes the child's stdout using the
            # PARENT's default locale encoding (cp1252 on this machine),
            # not UTF-8 — even though the child (with PYTHONIOENCODING=utf-8
            # below) is emitting UTF-8 bytes. Non-ASCII output (box-drawing
            # characters in evaluate.py's results table, "—" in device
            # names, etc.) then raises UnicodeDecodeError *inside this loop*,
            # which was an uncaught crash in run_pipeline.py itself —
            # reproducibly killing the pipeline at the same spot every time,
            # not an external interruption as it first appeared to be.
            t0 = time.time()
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            for out_line in proc.stdout:
                print(out_line, end="", flush=True)
                log.write(out_line)
                log.flush()
            returncode = proc.wait()
            elapsed = (time.time() - t0) / 60

            status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
            line = f"  -> {name}: {status}  ({elapsed:.1f} min)\n"
            print(line, flush=True)
            log.write(line)
            log.flush()

            if returncode != 0:
                print(f"\nPipeline stopped at '{name}'. See {log_path}", flush=True)
                sys.exit(returncode)

    total = (time.time() - t_start) / 3600
    print(f"\nPipeline complete in {total:.1f} hours. Full log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
