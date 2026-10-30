"""Memetic polishing of an evolved rule: structural local search, then a (1+lambda)-ES
on its numeric constants.

EoH's LLM mutations explore well but refine poorly: simply deleting parts of the
best R1 rule improved its fitness markedly (ablate.py). This stage does the
refinement systematically, on an objective that is harder to game than the
search fitness: every task is scored at its search horizon and at 2x that
horizon (rules tied to the exact step budget, e.g. hidden exponential lr decays,
lose at 2x), relative to LR-tuned Adam at the same horizon and seed, and the
per-task scores are aggregated with the robust fitness of prob.aggregate_fitness.

1. Structural phase: every edit of ablate.candidate_edits (drop a statement,
   unwrap a call, keep one side of an operation or branch) is scored; the best
   one is kept if it improves the objective by more than --margin; repeat.
2. Constant phase: a (1+lambda)-ES with the 1/5th success rule over the rule's
   numeric literals (except the structural 0, 0.5, 1 and 2), in log space,
   or in log(1 - c) for decay factors c in (0.5, 1).
The lr is re-tuned for every candidate locally (x1/3, x1, x3 around the lr of
the current best, per task and horizon), after a full tuning of the start rule.
Candidates are first ranked by a cheap proxy (search horizon only, at the current
lr); only the best --confirm of them are scored with the full objective.

    python analysis/polish.py --rule run:R1:1 --workers 8
    python analysis/polish.py --rule sample:R1:402 --name R1@402

Writes analysis_out/polish_<name>/{trace.json,<name>_polished.py}. Validate the result on
fresh seeds and the transfer suite with evaluation/transfer.py.
"""

import argparse
import ast
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ablate import candidate_edits
from common import ROOT, resolve_rules
from harness import check_rule, compile_rule, curve_score, run_rule, strip_docstrings, tune_lr
from prob import aggregate_fitness, reference_scores
from tasks import R3_SUITE, SEARCH_SUITE, make_task

STRUCTURAL = {0.0, 0.5, 1.0, 2.0}


def _tunable(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool) and abs(n.value) not in STRUCTURAL]


def constants(code):
    return [n.value for n in _tunable(ast.parse(code))]


def render(code, values):
    """code with its tunable constants (in ast.walk order) replaced by values."""
    tree = ast.parse(code)
    for node, v in zip(_tunable(tree), values):
        node.value = max(1, int(round(v))) if isinstance(node.value, int) else float(v)
    return ast.unparse(tree)


def encode(v):
    return math.log(1 - v) if 0.5 < v < 1 else math.log(abs(v))


def decode(z, like):
    if 0.5 < like < 1:  # decay factor: stay in (0.5, 1 - 1e-6]
        return 1 - math.exp(min(max(z, math.log(1e-6)), math.log(0.5)))
    return math.copysign(math.exp(min(max(z, -30.0), 30.0)), like)


def objective(code, tasks, scales, seeds, how, refs, lrs, lr_factors=(1 / 3, 1.0, 3.0), metric="train"):
    """(fitness, info) of a rule; lrs[key] is the lr to search around (None: full tuning)."""
    try:
        cls = compile_rule(code)
        if check_rule(cls) is not None:
            return None, None
        rel, new_lrs = {t: [] for t in tasks}, {}
        for seed in seeds:
            for sc in scales:
                for t in tasks:
                    task = make_task(t)
                    task.steps = max(20, int(round(task.steps * sc)))
                    key = f"{t}|{sc}|{seed}"
                    lr0 = (lrs or {}).get(key)
                    if lr0 is None:
                        res = tune_lr(task, cls, seed=seed, metric=metric)
                        score, lr = res["score"], res["lr"]
                    else:
                        score, lr = min((curve_score(run_rule(task, cls, lr0 * k, seed=seed, metric=metric)), lr0 * k)
                                        for k in lr_factors)
                    rel[t].append(score - refs[f"{sc}|{seed}"][t])
                    new_lrs[key] = lr
        rel = {t: float(np.mean(v)) for t, v in rel.items()}
        return aggregate_fitness(rel, how), dict(rel=rel, lrs=new_lrs)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, help="rule spec (see common.py)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--objective", choices=["mean", "robust"], default="robust")
    ap.add_argument("--tasks", nargs="+", default=None, help="default: the search suite (R3 suite with --metric val)")
    ap.add_argument("--metric", choices=["train", "val"], default="train")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 2.0], help="horizon multipliers")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rounds", type=int, default=3, help="max structural rounds")
    ap.add_argument("--gens", type=int, default=12, help="ES generations")
    ap.add_argument("--lam", type=int, default=12, help="ES offspring per generation")
    ap.add_argument("--confirm", type=int, default=4, help="candidates re-scored with the full objective")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.003, help="minimum improvement to accept")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.tasks = args.tasks or (R3_SUITE if args.metric == "val" else SEARCH_SUITE)
    rule = resolve_rules([args.rule])[0]
    name = args.name or rule["name"].replace("#", "_")
    out_dir = os.path.join(ROOT, "analysis_out", f"polish_{name}")
    os.makedirs(out_dir, exist_ok=True)
    refs = {f"{sc}|{seed}": reference_scores(args.tasks, seed=seed, steps_scale=sc, verbose=False,
                                             metric=args.metric)
            for seed in args.seeds for sc in args.scales}
    fixed = (args.tasks, args.scales, args.seeds, args.objective, refs)
    lr_factors = (1 / 3, 1.0, 3.0)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    rng = np.random.default_rng(0)
    t0 = time.time()

    def score_all(codes, lrs, proxy=False):
        if proxy:
            cheap = (args.tasks, args.scales[:1], args.seeds, args.objective, refs)
            futs = [pool.submit(objective, c, *cheap, lrs, (1.0,), args.metric) for c in codes]
        else:
            futs = [pool.submit(objective, c, *fixed, lrs, lr_factors, args.metric) for c in codes]
        return [f.result() for f in futs]

    def screen(codes, lrs):
        """Rank codes by the proxy, re-score the best args.confirm with the full
        objective; returns [(index, fitness, info)] best first."""
        prox = score_all(codes, lrs, proxy=True)
        top = sorted((i for i, r in enumerate(prox) if r[0] is not None), key=lambda i: prox[i][0])[:args.confirm]
        full = score_all([codes[i] for i in top], lrs)
        return sorted([(i, *r) for i, r in zip(top, full) if r[0] is not None], key=lambda e: e[1])

    code = ast.unparse(strip_docstrings(ast.parse(rule["code"])))
    fit, info = score_all([code], None)[0]
    if fit is None:
        raise SystemExit(f"start rule {rule['name']} is invalid")
    trace = [dict(phase="start", edit=rule["name"], fitness=fit, rel=info["rel"], code=code, time=0.0)]
    print(f"start {fit:+.4f} {info['rel']}", flush=True)

    def accept(phase, edit, f, inf, c):
        trace.append(dict(phase=phase, edit=edit, fitness=f, rel=inf["rel"], code=c, time=time.time() - t0))
        print(f"[{phase}] {edit[:80]} -> {f:+.4f} ({f - trace[-2]['fitness']:+.4f}) "
              f"{ {k: round(v, 3) for k, v in inf['rel'].items()} }", flush=True)

    for rnd in range(1, args.rounds + 1):
        edits = candidate_edits(code)
        ranked = screen([c for _, c in edits], info["lrs"]) if edits else []
        if not ranked or ranked[0][1] >= fit - args.margin:
            print(f"[structure] round {rnd}: no edit improves "
                  f"(best {ranked[0][1] if ranked else float('nan'):+.4f} of {len(edits)})", flush=True)
            break
        i, f, inf = ranked[0]
        code, fit, info = edits[i][1], f, inf
        accept("structure", edits[i][0], f, inf, code)

    values = constants(code)
    x = np.array([encode(v) for v in values])
    sigma = args.sigma
    print(f"[constants] {len(values)} tunable: {values}", flush=True)
    for gen in range(1, args.gens + 1 if values else 1):
        zs = [x + sigma * rng.standard_normal(len(x)) for _ in range(args.lam)]
        cands = [render(code, [decode(z, v) for z, v in zip(zv, values)]) for zv in zs]
        ranked = screen(cands, info["lrs"])
        if ranked and ranked[0][1] < fit - args.margin:
            i, fit, info = ranked[0]
            code = cands[i]
            accept("constants", f"gen {gen}, sigma {sigma:.3f}", fit, info, code)
            values = constants(code)
            x = np.array([encode(v) for v in values])
            sigma *= 1.4
        else:
            sigma *= 1.4 ** -0.25
            print(f"[constants] gen {gen}: no improvement (best {ranked[0][1] if ranked else float('nan'):+.4f}), "
                  f"sigma {sigma:.3f}", flush=True)
    pool.shutdown()

    with open(os.path.join(out_dir, "trace.json"), "w") as f:
        json.dump(dict(rule=rule["name"], args=vars(args), trace=trace), f, indent=2)
    with open(os.path.join(out_dir, f"{name}_polished.py"), "w") as f:
        f.write(f"# {rule['name']} polished ({args.objective} objective, horizons x{args.scales}, seeds "
                f"{args.seeds}): {trace[0]['fitness']:+.4f} -> {fit:+.4f}\n{code}\n")
    print(f"done in {(time.time() - t0) / 60:.0f} min: {trace[0]['fitness']:+.4f} -> {fit:+.4f}; wrote {out_dir}")


if __name__ == "__main__":
    main()
