"""L4 - how much of the natural gradient does an element-wise rule capture?

Part A (alignment): on a small classification task (tiny_fmnist, ~2k
parameters, or softmax_illcond, 650) form the exact Fisher F (= GGN) on a fixed
probe batch every `every` steps and record, along training,
    cos(delta, -(F + lam I)^-1 m)    alignment with full NGD
    cos(delta, -m / (diag F + lam))  alignment with diagonal NGD
    cos(diag NGD, full NGD)          diag_ref: how close the diagonal-NGD direction
                                     itself is to full NGD (a reference, not a bound)
    rhoF(delta)                      Fisher-cosine (proposition P2):
        rhoF(u) = -g^T u / (||u||_F * ||g||_{F^-1}),   g = probe-batch gradient,
        the first-order progress of u per unit of KL, relative to NGD; rhoF <= 1
        with equality iff u is parallel to -F^-1 g. Also reported for the
        reference directions -g (SGD), -g/sqrt(diag F), -g/diag F and -F^-1 g (= 1).
        The actual step delta is built from noisy minibatch gradients and
        momentum, so late in training rhoF(delta) is dominated by noise.
    rhoF_P                           rhoF of -P g with the rule's estimated diagonal
        preconditioner P = |delta| / |m| (clipped to its 1-99% quantiles): the
        geometry of the rule, separated from gradient noise. Exact for Adam,
        SGD-momentum, Signum and diag-NGD (all with beta1 = 0.9 like the probe).
    kl = delta^T F delta / 2         second-order KL change of the step
where m is a bias-corrected gradient EMA kept by the probe (beta = 0.9) and F is
damped by lam = damping * mean(diag F) in every solve and Fisher norm. Summaries
average the checkpoints of the whole run ("all"), its first or its second half.

Part B (rotation): curve scores on axis-aligned vs rotated ill-conditioned
softmax regression (same spectrum). Element-wise rules exploit axis alignment
and degrade under rotation; SGD and full NGD are rotation-equivariant.

    python analysis/ngd_alignment.py --rules baseline:sgd_momentum baseline:adam run:R1:3 --out a.json
    python analysis/ngd_alignment.py --all-samples R1 --task softmax_illcond --workers 6 --out r1.csv
"""

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as nnf

from common import code_id, load_eval_log, load_rule, load_samples, resolve_rules, valid_samples
from fisher import full_fisher
from harness import FlatModel, check_rule, default_device, run_rule, tune_lr
from tasks import make_task

METRICS = ("cos_full", "cos_diag", "cos_sgd", "diag_ref", "rhoF", "rhoF_P", "rhoF_sgd", "rhoF_adamF",
           "rhoF_diag", "rhoF_full", "kl")
SMALL_TASKS = ("tiny_fmnist", "softmax_illcond", "softmax_illcond_rot")


def make_full_ngd(task, refresh=10, probe=1024, damping=1e-3, beta=0.9, seed=0):
    """Reference full natural-gradient rule (exact Fisher, recomputed every `refresh` steps).

    Not element-wise: it closes over the task to evaluate F at the current parameters.
    """

    class FullNGD:
        def init_state(self, p):
            task.setup(p.device)
            gen = torch.Generator(device=p.device).manual_seed(seed + 555)
            bs = task.batch_size
            task.batch_size = probe
            x, _ = task.batch(gen)
            task.batch_size = bs
            return {"m": torch.zeros_like(p), "x": x, "flat": FlatModel(task.build().to(p.device))}

        def step(self, p, g, state, t, lr):
            m = state["m"]
            m.mul_(beta).add_(g, alpha=1 - beta)
            if t % refresh == 1 or refresh == 1:
                flat = state["flat"]
                flat.theta = p
                F = full_fisher(flat, state["x"]).double()
                lam = damping * F.diagonal().mean()
                L, info = torch.linalg.cholesky_ex(F + lam * torch.eye(F.shape[0], device=p.device, dtype=F.dtype))
                state["L"] = L if int(info) == 0 and bool(torch.isfinite(L).all()) else None
            if state["L"] is None:  # parameters blew up at this lr: report divergence
                return torch.full_like(p, float("nan"))
            nat = torch.cholesky_solve((m / (1 - beta ** t)).double()[:, None], state["L"]).squeeze(1)
            return -lr * nat.to(p.dtype)

    return FullNGD


def _cos(a, b):
    a, b = a.double(), b.double()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def alignment(rule_cls, fisher_sig, task, lr, every=20, probe=512, beta=0.9, damping=1e-3, seed=0):
    """Train once and return one dict of direction metrics per checkpoint (see module doc)."""
    device = default_device()
    task.setup(device)
    gen = torch.Generator(device=device).manual_seed(seed + 777)
    bs = task.batch_size
    task.batch_size = probe
    xp, yp = task.batch(gen)
    task.batch_size = bs
    ema, rows = {}, []

    def hook(t, info):
        ema["m"] = beta * ema.get("m", 0) + (1 - beta) * info["g"]
        if t % every:
            return
        flat = info["flat"]
        m = (ema["m"] / (1 - beta ** t)).double()
        F = full_fisher(flat, xp).double()
        lam = damping * F.diagonal().mean()
        F_lam = F + lam * torch.eye(F.shape[0], device=F.device, dtype=F.dtype)
        chol, bad = torch.linalg.cholesky_ex(F_lam)
        if int(bad) != 0 or not bool(torch.isfinite(chol).all()):
            return  # parameters are no longer finite: skip the checkpoint

        def solve(v):
            return torch.cholesky_solve(v[:, None], chol).squeeze(1)

        gp, = torch.autograd.grad(nnf.cross_entropy(flat(xp), yp), flat.theta)
        gp = gp.double()
        g_nat = solve(gp)
        g_norm = (gp @ g_nat).clamp_min(1e-300).sqrt()  # ||g||_{F^-1}

        def rho(u):
            return float(-(gp @ u) / ((u @ (F_lam @ u)).clamp_min(1e-300).sqrt() * g_norm))

        d = info["delta"].detach().double()
        d_full, d_diag = -solve(m), -m / F_lam.diagonal()
        P = d.abs() / m.abs().clamp_min(1e-30)
        lo, hi = torch.quantile(P[torch.isfinite(P)].float(), torch.tensor([0.01, 0.99], device=P.device))
        P = P.clamp(float(lo), float(hi))
        rows.append(dict(t=t, loss=float(info["loss"]),
                         cos_full=_cos(d, d_full), cos_diag=_cos(d, d_diag), cos_sgd=_cos(d, -m),
                         diag_ref=_cos(d_diag, d_full),
                         rhoF=rho(d), rhoF_P=rho(-P * gp), rhoF_sgd=rho(-gp),
                         rhoF_adamF=rho(-gp / F_lam.diagonal().sqrt()),
                         rhoF_diag=rho(-gp / F_lam.diagonal()), rhoF_full=rho(-g_nat),
                         kl=float(0.5 * d @ (F @ d))))

    run_rule(task, rule_cls, lr, seed=seed, use_fisher=fisher_sig, hook=hook)
    return rows


def phase_means(rows, phase="all"):
    """Mean of every metric over all checkpoints, the first or the second half (None if empty)."""
    half = len(rows) // 2
    sel = dict(all=rows, early=rows[:half], late=rows[half:])[phase]
    if not sel:
        return None
    return {k: float(np.nanmean([r[k] for r in sel])) for k in METRICS}


def _sample_job(rule, task_name, lr):
    """Late-run alignment metrics of one EoH sample (runs in a worker process)."""
    out = dict(sample_order=rule["sample_order"], code_id=rule["code_id"], operator=rule["operator"],
               objective=rule["objective"], task=task_name)
    try:
        cls, fisher = load_rule(rule)
        reason = check_rule(cls, fisher)
        if reason is not None:
            return dict(out, error=f"not element-wise: {reason}")
        task = make_task(task_name)
        if lr is None:  # task not in the search suite: cheap tuning (probe + one full run)
            lr = tune_lr(task, cls, use_fisher=fisher, refine=(1.0,))["lr"]
        means = phase_means(alignment(cls, fisher, task, lr))
        if means is None:
            return dict(out, lr=lr, error="no checkpoint (diverged early)")
        return dict(out, lr=lr, **means)
    except Exception as e:
        return dict(out, error=f"{type(e).__name__}: {e}"[:300])


def all_samples(run, task_name, workers, out, subsample=None):
    """Alignment metrics for every valid sample of a run (or a fixed random subset), written to a CSV."""
    eval_log = load_eval_log(run)
    jobs = []
    for s in valid_samples(run, subsample):
        rec = eval_log.get(s.get("code_id") or code_id(s["code"]), {})
        lr = rec.get("tasks", {}).get(task_name, {}).get("lr")  # search tasks: lr from the search
        jobs.append((dict(sample_order=s["sample_order"], code_id=s["code_id"], operator=s["operator"],
                          objective=s["objective"], code=s["code"]), task_name, lr))
    print(f"{len(jobs)} valid samples on {task_name}", flush=True)
    if not jobs:
        return
    rows = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, row in enumerate(pool.map(_sample_job, *zip(*jobs))):
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    fields = ["sample_order", "code_id", "operator", "objective", "task", "lr", *METRICS, "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum("error" not in r for r in rows)
    print(f"wrote {out} ({n_ok}/{len(rows)} ok)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="+", default=["tiny_fmnist", "softmax_illcond"], choices=SMALL_TASKS,
                    help="alignment tasks for --rules (exact Fisher: small models only)")
    ap.add_argument("--no-full-ngd", action="store_true", help="skip the full-NGD reference rule")
    ap.add_argument("--skip-rotation", action="store_true")
    ap.add_argument("--all-samples", default=None, help="run name: alignment of every valid sample")
    ap.add_argument("--task", default="softmax_illcond", choices=SMALL_TASKS, help="task for --all-samples")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subsample", type=int, default=None, help="--all-samples: fixed random subset of n samples")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.all_samples:
        all_samples(args.all_samples, args.task, args.workers,
                    args.out or f"{args.all_samples}_{args.task}_alignment.csv", args.subsample)
        return

    results = {}
    rules = [(r["name"], *load_rule(r)) for r in resolve_rules(args.rules)]
    entries = rules + ([] if args.no_full_ngd else [("full_ngd", None, False)])

    for task_name in args.tasks:
        for name, cls, fisher in entries:
            task = make_task(task_name)
            rule_cls = cls or make_full_ngd(task)
            lr = tune_lr(task, rule_cls, use_fisher=fisher)["lr"]
            rows = alignment(rule_cls, fisher, task, lr)
            means = {ph: phase_means(rows, ph) for ph in ("all", "early", "late")}
            results.setdefault(name, {}).setdefault("alignment", {})[task_name] = dict(lr=lr, rows=rows, means=means)
            a = means["all"]
            if a is None:
                print(f"{name:<18} {task_name}: no checkpoint (diverged early)", flush=True)
                continue
            print(f"{name:<18} {task_name} lr={lr:.2g}  rhoF_P={a['rhoF_P']:.3f} rhoF={a['rhoF']:+.3f}  "
                  f"cos(full NGD)={a['cos_full']:+.3f}  cos(diag NGD)={a['cos_diag']:+.3f}  "
                  f"cos(SGD)={a['cos_sgd']:+.3f}  diag_ref={a['diag_ref']:+.3f}  [rhoF: sgd={a['rhoF_sgd']:.3f} "
                  f"adamF={a['rhoF_adamF']:.3f} diag={a['rhoF_diag']:.3f} full={a['rhoF_full']:.3f}]",
                  flush=True)

    if not args.skip_rotation:
        aligned, rotated = make_task("softmax_illcond"), make_task("softmax_illcond_rot")
        for name, cls, fisher in entries:
            scores = {}
            for task in (aligned, rotated):
                rule_cls = cls or make_full_ngd(task)
                scores[task.name] = tune_lr(task, rule_cls, use_fisher=fisher)["score"]
            gap = scores[rotated.name] - scores[aligned.name]
            print(f"{name:<18} aligned={scores[aligned.name]:+.3f}  rotated={scores[rotated.name]:+.3f}  "
                  f"rotation penalty={gap:+.3f}", flush=True)
            results.setdefault(name, {})["rotation"] = dict(scores, penalty=gap)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
