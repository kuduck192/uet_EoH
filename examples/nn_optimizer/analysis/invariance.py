"""P1 - reparametrization equivariance of update rules.

Under a diagonal reparametrization phi = D * theta, gradients transform as
g_phi = g / D (so do Fisher samples) and the diagonal Fisher as F / D**2.
A rule whose update transforms as delta_phi = D**k * delta_theta has
equivariance exponent k:

    SGD k = -1,   Adam / sign methods k = 0,   diagonal natural gradient k = +1.

Only k = +1 gives the same trajectory in both coordinate systems. For a rule
delta ~ m**b * F**(-alpha), k = 2*alpha - b (cross-checked by alpha_fit.py).

Two measurements:
  rule-level  k(D) on synthetic gradient streams: no training, milliseconds,
              cheap enough to run on every sample of an EoH run;
  training    best loss-curve score in theta vs in phi = D * theta coordinates
              (per-coordinate log-uniform D, spread 10 and 100), each with its own
              tuned lr, so only the relative distortion of coordinates matters.

    python analysis/invariance.py --rules baseline:sgd_momentum baseline:adam \
        baseline:diag_ngd_emp run:R1:3 --task softmax_illcond
"""

import argparse
import json
import math

import torch

from common import load_rule, resolve_rules
from harness import default_device, tune_lr
from tasks import make_task


def _stream(n, T, seed, device, log10_range=(-1.0, 1.0)):
    """Synthetic per-coordinate gradient streams with scales log-uniform in 10**log10_range.

    The default (0.1 .. 10) keeps g*g far above the absolute damping / eps
    constants of the baselines, so rule_exponent measures the scale-free exponent
    of a rule; alpha_fit.py measures the operational one on real gradients.
    """
    gen = torch.Generator().manual_seed(seed)
    lo, hi = log10_range
    scale = 10.0 ** (lo + torch.rand(n, generator=gen) * (hi - lo))
    mean = 0.5 * scale * torch.randn(n, generator=gen)
    p = 0.1 * torch.randn(n, generator=gen)
    seq = []
    for _ in range(T):
        seq.append(dict(p=p.to(device), g=(mean + scale * torch.randn(n, generator=gen)).to(device),
                        f=(scale * torch.randn(n, generator=gen)).to(device)))
    return seq


def _run(rule_cls, fisher, seq, D, lr):
    rule = rule_cls()
    state = rule.init_state(seq[0]["p"] * D)
    for t, s in enumerate(seq, start=1):
        args = [s["p"] * D, s["g"] / D] + ([s["f"] / D] if fisher else [])
        delta = rule.step(*args, state, t, lr)
    return delta.double()


def rule_exponent(rule_cls, fisher, D, n=20000, T=50, lr=1e-3, seed=0, device=None,
                  log10_range=(-1.0, 1.0)):
    """Per-coordinate k = log(|delta_phi| / |delta_theta|) / log D after T steps."""
    device = device or default_device()
    seq = _stream(n, T, seed, device, log10_range)
    d_theta = _run(rule_cls, fisher, seq, 1.0, lr)
    d_phi = _run(rule_cls, fisher, seq, D, lr)
    ok = torch.isfinite(d_theta) & torch.isfinite(d_phi) & (d_theta != 0) & (d_phi != 0)
    if ok.sum() == 0:
        return dict(D=D, median=float("nan"), q25=float("nan"), q75=float("nan"), frac_valid=0.0)
    k = (d_phi[ok].abs().log() - d_theta[ok].abs().log()) / math.log(D)
    q = torch.quantile(k.float().cpu(), torch.tensor([0.25, 0.5, 0.75]))
    return dict(D=D, median=float(q[1]), q25=float(q[0]), q75=float(q[2]),
                frac_valid=float(ok.float().mean()))


def training_gap(rule_cls, fisher, task, spread=10.0, seed=0):
    """Best curve_score in theta vs phi = D * theta coordinates (D_i log-uniform in
    [1/spread, spread]), each with its own tuned lr, so only the relative distortion
    of the coordinates matters. A rule with k = +1 is unaffected (up to damping)."""
    def scale(n, device):
        gen = torch.Generator().manual_seed(seed + 99)
        return torch.exp((torch.rand(n, generator=gen) * 2 - 1) * math.log(spread)).to(device)

    theta = tune_lr(task, rule_cls, seed=seed, use_fisher=fisher)
    phi = tune_lr(task, rule_cls, seed=seed, use_fisher=fisher, param_scale=scale)
    return dict(spread=spread, lr_theta=theta["lr"], lr_phi=phi["lr"], score_theta=theta["score"],
                score_phi=phi["score"], gap=phi["score"] - theta["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--D", nargs="+", type=float, default=[0.01, 0.1, 10.0, 100.0])
    ap.add_argument("--task", default=None, help="also run the training test on this task")
    ap.add_argument("--spreads", nargs="+", type=float, default=[10.0, 100.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for rule in resolve_rules(args.rules):
        cls, fisher = load_rule(rule)
        res = dict(rule_level=[rule_exponent(cls, fisher, D) for D in args.D])
        line = f"{rule['name']:<18} k(D): " + "  ".join(
            f"D={r['D']:g}:{r['median']:+.2f}[{r['q25']:+.2f},{r['q75']:+.2f}]" for r in res["rule_level"])
        if args.task:
            res["training"] = [training_gap(cls, fisher, make_task(args.task), sp) for sp in args.spreads]
            line += f"  | {args.task}: " + "  ".join(
                f"spread {tr['spread']:g}: theta={tr['score_theta']:.3f} phi={tr['score_phi']:.3f} "
                f"gap={tr['gap']:+.3f}" for tr in res["training"])
        print(line, flush=True)
        results[rule["name"]] = res
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
