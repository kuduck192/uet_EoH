"""Re-evaluate rules with several seeds on the search suite or on held-out tasks.

For each (rule, task): tune lr with seed 0 (same procedure as the search),
then train seeds 1..S at that lr (disjoint from the search seed 0). Reported
per task, relative to Adam tuned identically (negative = faster than Adam):
    rel      curve_score(rule) - curve_score(Adam)  paired by seed, mean +- std
    final    final smoothed training loss of the seed-averaged curve
    speedup  Adam's steps to its own final loss / the rule's steps to that loss,
             both on seed-averaged curves (>1 = faster; None = never reached)

    python evaluation/transfer.py --rules baseline:lion run:R1:3 --suite transfer --seeds 3 --workers 4
"""

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

from common import ROOT, resolve_rules, uses_fisher  # noqa: E402
from baselines import ADAM  # noqa: E402
from harness import compile_rule, curve_score, run_rule, tune_lr  # noqa: E402
from tasks import R3_SUITE, R3_TRANSFER_SUITE, SEARCH_SUITE, TRANSFER_SUITE, make_task  # noqa: E402


def _smooth(losses, w=20):
    losses = np.asarray(losses, float)
    if not np.isfinite(losses).all():
        losses = losses.copy()
        losses[np.argmax(~np.isfinite(losses)):] = np.inf
    kernel = np.ones(w) / w
    return np.convolve(losses, kernel, mode="valid")


def steps_to(losses, target, w=20):
    """First step at which the smoothed loss reaches target (inf if never)."""
    s = _smooth(losses, w)
    hit = np.nonzero(s <= target)[0]
    return float(hit[0] + w) if hit.size else float("inf")


def mean_curve(curves):
    """Seed-averaged loss curve; a seed that diverged counts as inf from then on."""
    arr = np.array([np.asarray(c, float) for c in curves])
    for row in arr:
        bad = ~np.isfinite(row)
        if bad.any():
            row[np.argmax(bad):] = np.inf
    return arr.mean(0)


def evaluate(name, code, task_name, n_seeds, seed_offset, metric="train"):
    try:
        cls = compile_rule(code)
        fisher = uses_fisher(cls)
        task = make_task(task_name)
        tuned = tune_lr(task, cls, seed=0, use_fisher=fisher, metric=metric)
        seeds = list(range(seed_offset, seed_offset + n_seeds))
        curves = [run_rule(task, cls, tuned["lr"], seed=s, use_fisher=fisher, metric=metric) for s in seeds]
        return dict(rule=name, task=task_name, lr=tuned["lr"], seeds=seeds,
                    scores=[curve_score(c) for c in curves], curves=[c.tolist() for c in curves])
    except Exception as e:
        return dict(rule=name, task=task_name, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--suite", default="transfer", help="search | transfer | comma-separated task names")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=1,
                    help="first evaluation seed; 1 keeps them disjoint from the search / lr-tuning seed 0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", choices=["train", "val"], default="train",
                    help="val: held-out loss curves (every 10 steps) of the *_v tasks")
    ap.add_argument("--merge-into", default=None,
                    help="existing output of this script: reuse its runs (incl. adam_ref), add/replace --rules")
    args = ap.parse_args()

    tasks = dict(search=SEARCH_SUITE, transfer=TRANSFER_SUITE, r3=R3_SUITE,
                 r3_transfer=R3_TRANSFER_SUITE).get(args.suite) or args.suite.split(",")
    w = 3 if args.metric == "val" else 20  # smoothing window, in curve points
    new_rules = resolve_rules(args.rules)
    previous = []
    table_tasks = list(tasks)
    if args.merge_into:
        with open(args.merge_into) as f:
            old = json.load(f)
        # keep the old columns (a merge may run the new rules on a subset of the tasks)
        table_tasks = list(next(iter(old["table"].values())).keys())
        table_tasks += [t for t in tasks if t not in table_tasks]
        new_names = {r["name"] for r in new_rules}
        previous = [r for r in old["raw"] if r["rule"] not in new_names]
        names = [n for n in old["table"] if n not in new_names] + [r["name"] for r in new_rules]
        rules = [dict(name=n) for n in names]
        to_run = new_rules
    else:
        rules = [dict(name="adam_ref", code=ADAM)] + new_rules
        to_run = rules
    jobs = [(r["name"], r["code"], t, args.seeds, args.seed_offset, args.metric) for r in to_run for t in tasks]

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = previous + list(pool.map(evaluate, *zip(*jobs)))
    by = {(r["rule"], r["task"]): r for r in results}
    for r in results:
        if "error" in r:
            print(f"[error] {r['rule']} on {r['task']}: {r['error']}", flush=True)

    table = {}
    for r in rules:
        row = {}
        for t in table_tasks:
            if (r["name"], t) not in by:
                row[t] = dict(error="not run")
                continue
            res, ref = by[(r["name"], t)], by[("adam_ref", t)]
            if "error" in res or "error" in ref:
                row[t] = dict(error=res.get("error") or ref.get("error"))
                continue
            rel = np.array(res["scores"]) - np.array(ref["scores"])  # paired by seed
            ref_curve, rule_curve = mean_curve(ref["curves"]), mean_curve(res["curves"])
            target = _smooth(ref_curve, w)[-1]  # Adam's final (seed-averaged, smoothed) loss
            adam_steps, rule_steps = steps_to(ref_curve, target, w), steps_to(rule_curve, target, w)
            final = _smooth(rule_curve, w)[-1]
            row[t] = dict(rel_mean=float(rel.mean()), rel_std=float(rel.std()), lr=res["lr"],
                          final=float(final) if np.isfinite(final) else None,
                          n_diverged=int(sum(not np.isfinite(c).all() for c in map(np.asarray, res["curves"]))),
                          speedup=float(adam_steps / rule_steps) if np.isfinite(rule_steps) else None)
        table[r["name"]] = row

    header = f"{'rule':<16}" + "".join(f"{t[:22]:>26}" for t in table_tasks)
    print(header)
    for name, row in table.items():
        cells = []
        for t in table_tasks:
            c = row[t]
            if "error" in c:
                cells.append(f"{c['error'] if c['error'] == 'not run' else 'error':>26}")
            else:
                sp = f"x{c['speedup']:<5.2f}" if c["speedup"] is not None else "x  -  "
                cells.append(f"{c['rel_mean']:>+9.3f}±{c['rel_std']:.3f} {sp}    ")
        print(f"{name:<16}" + "".join(cells))

    out = args.out or args.merge_into or os.path.join(ROOT, "analysis_out", f"eval_{args.suite.replace(',', '+')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(table=table, raw=results, metric=args.metric), f)
    print("wrote", out)


if __name__ == "__main__":
    main()
