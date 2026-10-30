"""L2 - implied preconditioner: where does a rule sit between SGD, Adam and diagonal NGD?

During training, every `every` steps, regress over coordinates

    log|delta_i| = c + b * log|m_i| - alpha * log F_ii

where m is a bias-corrected EMA of the gradient kept by the probe (beta=0.9,
independent of the rule's own state) and F_ii is one of three curvature
statistics: the exact diagonal of the true Fisher (= GGN for softmax CE) or of
the empirical Fisher on a fixed probe batch, or "gsq", the EMA of the squared
minibatch gradient (what Adam-like rules actually observe). Comparing alpha
across sources separates "what statistic the rule is a function of" from "how
that statistic relates to the Fisher". Reference points:

    SGD (b, alpha) = (1, 0)    Adam ~ (1, 1/2)    sign methods ~ (0, 0)    diag NGD = (1, 1)

and the implied equivariance exponent is k = 2*alpha - b (compare invariance.py).

    python analysis/alpha_fit.py --rules baseline:adam run:R2:3 --task mlp_fmnist
    python analysis/alpha_fit.py --all-samples R1 --task softmax_illcond --out r1_alpha.csv
"""

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from common import load_eval_log, load_rule, load_samples, load_seeds, resolve_rules, valid_samples
from fisher import diag_fisher, mc_diag_fisher
from harness import check_elementwise, default_device, run_rule, tune_lr
from invariance import rule_exponent
from tasks import CharLM, make_task

MAX_COORDS = 200_000


def regress(delta, m, Fd):
    """Fit log|delta| = c + b log|m| - alpha log F. Returns dict(alpha, b, r2, alpha_b1, n)."""
    delta, m, Fd = delta.double(), m.double(), Fd.double()
    ok = torch.isfinite(delta) & torch.isfinite(m) & (delta != 0) & (m != 0) & (Fd > 0)
    idx = ok.nonzero().squeeze(1)
    if idx.numel() < 16:
        return None
    if idx.numel() > MAX_COORDS:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:MAX_COORDS]]
    y = delta[idx].abs().log()
    lm, lf = m[idx].abs().log(), Fd[idx].log()
    X = torch.stack([torch.ones_like(y), lm, lf], 1)
    sol = torch.linalg.lstsq(X, y[:, None]).solution.squeeze(1)
    resid = y - X @ sol
    r2 = 1 - resid.var() / y.var().clamp_min(1e-30)
    # b fixed to 1: log(|delta| / |m|) = c - alpha log F
    X1 = torch.stack([torch.ones_like(y), lf], 1)
    sol1 = torch.linalg.lstsq(X1, (y - lm)[:, None]).solution.squeeze(1)
    return dict(alpha=float(-sol[2]), b=float(sol[1]), r2=float(r2), alpha_b1=float(-sol1[1]),
                n=int(idx.numel()))


def _cos(a, b):
    a, b = a.double(), b.double()
    ok = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[ok], b[ok]
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def fit_rule(rule_cls, fisher_sig, task, lr, steps=None, every=25, probe_size=512, beta=0.9,
             seed=0, by_layer=True, device=None):
    """Train once and return dict(records=[per-checkpoint fits], losses=[...])."""
    device = device or default_device()
    task.setup(device)
    gen = torch.Generator(device=device).manual_seed(seed + 777)
    bs = task.batch_size
    task.batch_size = probe_size if not isinstance(task, CharLM) else min(bs, 32)
    xp, yp = task.batch(gen)
    task.batch_size = bs
    sequence = isinstance(task, CharLM)

    ema = {}
    records = []

    def hook(t, info):
        g = info["g"]
        # Zero-initialised EMAs with bias correction below, exactly as in Adam.
        ema["m"] = beta * ema.get("m", 0) + (1 - beta) * g
        ema["v"] = 0.999 * ema.get("v", 0) + 0.001 * g * g
        if t % every:
            return
        m_hat = ema["m"] / (1 - beta ** t)
        flat, delta = info["flat"], info["delta"].detach()
        # gsq: EMA of the squared minibatch gradient, the statistic Adam-like rules
        # actually see; by P2 it mixes diag F_emp / B with the squared mean gradient.
        sources = dict(gsq=ema["v"] / (1 - 0.999 ** t))
        if sequence:
            sources["true"] = mc_diag_fisher(flat, task, xp, yp, gen)
        else:
            sources["true"], sources["emp"] = diag_fisher(flat, xp, yp)
        rec = dict(t=t, loss=float(info["loss"]))
        for name, Fd in sources.items():
            rec[name] = regress(delta, m_hat, Fd)
            if by_layer:
                rec[name + "_layers"] = {
                    layer: regress(delta[s], m_hat[s], Fd[s])
                    for layer, s in flat.slices().items() if s.stop - s.start >= 64
                }
            lam = 1e-4 * Fd.mean()
            rec[name + "_cos"] = dict(sgd=_cos(delta, -m_hat),
                                      adam=_cos(delta, -m_hat / (Fd.sqrt() + lam.sqrt())),
                                      ngd=_cos(delta, -m_hat / (Fd + lam)))
        records.append(rec)

    losses = run_rule(task, rule_cls, lr, steps, seed, fisher_sig, device, hook=hook)
    return dict(lr=lr, records=records, losses=losses.tolist())


def summarize(fit, source="true", warmup_frac=0.25):
    """Median (alpha, b, k = 2 alpha - b) over checkpoints after warm-up."""
    recs = [r for r in fit["records"] if r.get(source)]
    if not recs:
        return dict(alpha=float("nan"), b=float("nan"), k=float("nan"), r2=float("nan"))
    recs = recs[int(len(recs) * warmup_frac):] or recs
    alpha = float(np.median([r[source]["alpha"] for r in recs]))
    b = float(np.median([r[source]["b"] for r in recs]))
    return dict(alpha=alpha, b=b, k=2 * alpha - b, r2=float(np.median([r[source]["r2"] for r in recs])),
                alpha_b1=float(np.median([r[source]["alpha_b1"] for r in recs])))


def _sample_job(rule, task_name, lr, every):
    """(alpha, b, k) fits of one EoH sample on task_name (runs in a worker process)."""
    out = dict(name=rule["name"], sample_order=rule["sample_order"], code_id=rule.get("code_id"),
               operator=rule.get("operator", "seed"), objective=rule["objective"])
    try:
        cls, fisher = load_rule(rule)
        reason = check_elementwise(cls, fisher)
        if reason is not None:
            return dict(out, error=f"not element-wise: {reason}")
        task = make_task(task_name)
        if lr is None:
            lr = tune_lr(task, cls, use_fisher=fisher)["lr"]
        fit = fit_rule(cls, fisher, task, lr, every=every, by_layer=False)
        s_true, s_emp, s_gsq = (summarize(fit, src) for src in ("true", "emp", "gsq"))
        return dict(out, lr=lr, alpha_true=s_true["alpha"], b_true=s_true["b"], k_fit=s_true["k"],
                    r2_true=s_true["r2"], alpha_emp=s_emp["alpha"], b_emp=s_emp["b"],
                    alpha_gsq=s_gsq["alpha"], b_gsq=s_gsq["b"], r2_gsq=s_gsq["r2"],
                    k_rule=rule_exponent(cls, fisher, 10.0)["median"])
    except Exception as e:  # rules that crash under the probe are reported, not fatal
        return dict(out, error=f"{type(e).__name__}: {e}"[:300])


FIELDS = ["name", "sample_order", "code_id", "operator", "objective", "lr", "alpha_true", "b_true", "k_fit",
          "r2_true", "alpha_emp", "b_emp", "alpha_gsq", "b_gsq", "r2_gsq", "k_rule", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="*", default=[])
    ap.add_argument("--all-samples", default=None, help="fit every valid sample (and seed) of this run")
    ap.add_argument("--task", default="mlp_fmnist")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=None, help="override the tuned learning rate")
    ap.add_argument("--workers", type=int, default=6, help="processes for --all-samples")
    ap.add_argument("--subsample", type=int, default=None, help="--all-samples: fixed random subset of n samples")
    ap.add_argument("--out", default=None, help=".json (per-rule fits) or .csv (with --all-samples)")
    args = ap.parse_args()

    if args.all_samples:
        run = args.all_samples
        eval_log = load_eval_log(run)
        rules = [dict(name=f"seed:{i}", code=s["code"], sample_order=0, objective=None)
                 for i, s in enumerate(load_seeds(run))]
        rules += [dict(name=f"#{s['sample_order']}", code=s["code"], sample_order=s["sample_order"],
                       objective=s["objective"], code_id=s["code_id"], operator=s["operator"])
                  for s in valid_samples(run, args.subsample)]
        # lr found by the search for this task (search-suite tasks), else tuned in the worker
        lrs = [args.lr or eval_log.get(r.get("code_id", ""), {}).get("tasks", {}).get(args.task, {}).get("lr")
               for r in rules]
        if not rules:
            print("no valid samples")
            return
        rows = []
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_sample_job, rules, [args.task] * len(rules), lrs, [args.every] * len(rules)):
                rows.append(row)
                print(", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
        out = args.out or f"{run}_{args.task}_alpha.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out} ({sum('error' not in r for r in rows)}/{len(rows)} ok)")
        return

    results = {}
    for rule in resolve_rules(args.rules):
        cls, fisher = load_rule(rule)
        task = make_task(args.task)
        lr = args.lr or tune_lr(task, cls, use_fisher=fisher)["lr"]
        fit = fit_rule(cls, fisher, task, lr, every=args.every)
        results[rule["name"]] = fit
        line = f"{rule['name']:<18} lr={lr:.2g}"
        for src in ("gsq", "true", "emp"):
            s = summarize(fit, src)
            line += f"  | {src}: alpha={s['alpha']:+.2f} b={s['b']:+.2f} k={s['k']:+.2f} R2={s['r2']:.2f}"
        print(line, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
