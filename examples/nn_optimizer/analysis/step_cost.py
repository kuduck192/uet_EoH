"""Wall-clock cost per training step of several rules.

Fitness and speedups count optimizer *steps*; this measures what one step costs in
seconds (forward + backward + update), so that step counts can be converted into
time. Run it on an otherwise idle GPU: concurrent jobs distort the timings.

    python analysis/step_cost.py --rules baseline:adam_t baseline:shampoo_cosine evolved:R4:1 \
        --tasks cnn_cifar_v gpt_v --steps 60 --out analysis_out/step_cost.json
"""

import argparse
import json
import time

import torch

from common import load_rule, resolve_rules
from harness import default_device, run_rule
from tasks import make_task


def time_rule(task_name, rule_cls, steps, lr=1e-3):
    """Seconds per step of rule_cls on task_name, after a short warm-up run."""
    device = default_device()
    task = make_task(task_name)
    run_rule(task, rule_cls, lr, steps=3, device=device)  # CUDA context, cuDNN autotuning, data on device
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_rule(task, rule_cls, lr, steps=steps, device=device)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["cnn_cifar_v", "gpt_v"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for task_name in args.tasks:
        for rule in resolve_rules(args.rules):
            cls, _ = load_rule(rule)
            sec = time_rule(task_name, cls, args.steps)
            results.setdefault(rule["name"], {})[task_name] = sec
            print(f"{rule['name']:<18} {task_name:<14} {1000 * sec:8.1f} ms/step", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
