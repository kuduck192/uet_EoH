"""L1 (optional) - symbolic regression of a rule's update as a function of
per-coordinate statistics, for rules whose code is too tangled to read.

Records (g, m, v, [f, F], p, t) -> delta / lr on random coordinates during
training, where m, v (and F) are bias-corrected EMAs of g, g*g (and f*f) kept by
the probe. Saves the samples as .npz and, if PySR is installed, fits closed
forms over these statistics.

    python analysis/symreg.py --rule run:R2:1 --task mlp_fmnist
"""

import argparse
import os

import numpy as np
import torch

from common import ROOT, load_rule, resolve_rules
from harness import default_device, run_rule, tune_lr
from tasks import make_task


def collect(rule_cls, fisher, task, lr, n_coords=2000, every=10, seed=0):
    device = default_device()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ema, rows, idx = {}, [], {}

    def hook(t, info):
        g, f = info["g"], info["f"]
        ema["m"] = 0.9 * ema.get("m", 0) + 0.1 * g
        ema["v"] = 0.999 * ema.get("v", 0) + 0.001 * g * g
        if f is not None:
            ema["F"] = 0.999 * ema.get("F", 0) + 0.001 * f * f
        if "i" not in idx:
            idx["i"] = torch.randperm(g.numel(), generator=gen)[:n_coords].to(device)
        if t % every:
            return
        i = idx["i"]
        cols = [g[i], ema["m"][i] / (1 - 0.9 ** t), ema["v"][i] / (1 - 0.999 ** t)]
        if f is not None:
            cols += [f[i], ema["F"][i] / (1 - 0.999 ** t)]
        cols += [info["theta"][i], torch.full_like(g[i], float(t)), info["delta"][i] / lr]
        rows.append(torch.stack(cols, 1).double().cpu().numpy())

    run_rule(task, rule_cls, lr, seed=seed, use_fisher=fisher, device=device, hook=hook)
    names = ["g", "m", "v"] + (["f", "F"] if fisher else []) + ["p", "t", "target"]
    data = np.concatenate(rows)
    return names, data[np.isfinite(data).all(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--task", default="mlp_fmnist")
    ap.add_argument("--iterations", type=int, default=60)
    args = ap.parse_args()

    rule = resolve_rules([args.rule])[0]
    cls, fisher = load_rule(rule)
    task = make_task(args.task)
    lr = tune_lr(task, cls, use_fisher=fisher)["lr"]
    names, data = collect(cls, fisher, task, lr)
    out_dir = os.path.join(ROOT, "analysis_out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"symreg_{rule['name'].replace('#', '_')}_{args.task}.npz")
    np.savez(path, data=data, names=np.array(names), lr=lr)
    print(f"saved {data.shape[0]} samples to {path}")

    try:
        from pysr import PySRRegressor
    except ImportError:
        print("PySR not installed (pip install pysr); fit the .npz offline.")
        return
    X, y = data[:, :-1], data[:, -1]
    sub = np.random.default_rng(0).choice(len(y), size=min(len(y), 5000), replace=False)
    model = PySRRegressor(niterations=args.iterations, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["sqrt", "abs", "sign", "tanh", "exp", "log"],
                          maxsize=25, model_selection="best")
    model.fit(X[sub], y[sub], variable_names=names[:-1])
    print(model)


if __name__ == "__main__":
    main()
