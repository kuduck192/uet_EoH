"""Population-level analysis: is EoH fitness associated with NGD-likeness?

Every valid sample of a run is one optimizer; together they form a natural
experiment. Per sample (keyed by sample_order) this merges the EoH fitness, the
per-task relative scores (eval log), motif tags and k_rule (motifs.py), the
(alpha, b, k_fit) fits (alpha_fit.py --all-samples) and the Fisher-alignment
metrics (ngd_alignment.py --all-samples), then reports Spearman rank
correlations with 2000-resample bootstrap 95% intervals. Samples share ancestry,
so they are not independent: the correlations are descriptive, not tests.

    python analysis/population.py R1 --alpha analysis_out/R1_softmax_illcond_alpha.csv \
        --align analysis_out/R1_softmax_illcond_alignment.csv analysis_out/R1_tiny_fmnist_alignment.csv

Writes analysis_out/population_corr.json and report/figures/pop_*.png.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import plotstyle
from common import ROOT, load_eval_log, run_dir
from tasks import TASKS

MOTIF_COLS = ["second_moment", "sqrt", "sign", "saturation", "bias_correction", "t_schedule", "uses_p", "exp_log"]


def _task_of(path):
    m = re.search(r"_(" + "|".join(sorted(TASKS, key=len, reverse=True)) + r")_alpha\.csv$", path)
    if not m:
        raise ValueError(f"cannot infer the task from {path} (expected <run>_<task>_alpha.csv)")
    return m.group(1)


def load_population(run, alpha_paths, align_paths):
    df = pd.read_csv(os.path.join(run_dir(run), "analysis", "samples.csv"))
    df = df[df["objective"].notna()].copy()
    rel = {cid: rec.get("relative", {}) for cid, rec in load_eval_log(run).items()}
    for task in sorted({t for r in rel.values() for t in r}):
        df[f"rel:{task}"] = df["code_id"].map(lambda c: rel.get(c, {}).get(task, np.nan))
    for path in alpha_paths:
        task = _task_of(path)
        a = pd.read_csv(path)
        if "error" in a:
            a = a[a["error"].isna()]
        keep = ["alpha_true", "b_true", "k_fit", "r2_true", "alpha_gsq", "b_gsq"]
        a = a[["sample_order"] + keep].rename(columns={k: f"{task}:{k}" for k in keep})
        df = df.merge(a, on="sample_order", how="left")
    for path in align_paths:
        a = pd.read_csv(path)
        if "error" in a:
            a = a[a["error"].isna()]
        task = a["task"].iloc[0] if len(a) else os.path.basename(path)
        keep = ["rhoF_P", "rhoF", "cos_full", "cos_diag", "kl"]
        a = a[["sample_order"] + keep].rename(columns={k: f"{task}:{k}" for k in keep})
        df = df.merge(a, on="sample_order", how="left")
    return df


def spearman_ci(x, y, n_boot=2000, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return dict(n=int(len(x)), rho=np.nan, lo=np.nan, hi=np.nan)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            boots.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=int(len(x)), rho=float(rho), lo=float(lo), hi=float(hi))


def family(row):
    if row.get("sign") and not row.get("sqrt"):
        return "sign-based"
    if row.get("sqrt") and row.get("second_moment"):
        return "√(second moment), Adam-like"
    return "other"


FAMILIES = ["√(second moment), Adam-like", "sign-based", "other"]


def scatter(df, xcol, xlabel, path, title):
    import matplotlib.pyplot as plt

    plotstyle.apply()
    d = df[np.isfinite(df[xcol]) & np.isfinite(df["objective"])]
    if d.empty:
        return False
    ycap = np.percentile(d["objective"], 95)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, fam in enumerate(FAMILIES):
        s = d[d["family"] == fam]
        if len(s):
            ax.scatter(s[xcol], s["objective"].clip(upper=ycap), s=22, color=plotstyle.SERIES[i],
                       edgecolor=plotstyle.SURFACE, linewidth=1.0, alpha=0.85, label=f"{fam} (n={len(s)})")
    best = d.loc[d["objective"].idxmin()]
    ax.scatter([best[xcol]], [best["objective"]], s=90, facecolor="none", edgecolor=plotstyle.INK, linewidth=1.4)
    ax.annotate("best", (best[xcol], best["objective"]), xytext=(6, -4), textcoords="offset points",
                fontsize=8, color=plotstyle.INK2)
    ax.axhline(0, color=plotstyle.INK2, lw=0.8)
    ax.text(ax.get_xlim()[1], 0, " Adam", va="center", ha="left", fontsize=8, color=plotstyle.INK2)
    r = spearman_ci(d[xcol].to_numpy(float), d["objective"].to_numpy(float))
    ax.set_title(f"{title}\nSpearman ρ = {r['rho']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], n = {r['n']}",
                 fontsize=9, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EoH fitness vs Adam\n(lower = better; clipped at p95)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    plotstyle.save(fig, path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--alpha", nargs="*", default=[])
    ap.add_argument("--align", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis_out", "population_corr.json"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "report", "figures"))
    args = ap.parse_args()

    df = load_population(args.run, args.alpha, args.align)
    df["family"] = df.apply(family, axis=1)
    os.makedirs(args.fig_dir, exist_ok=True)

    targets = ["objective"] + [c for c in df if c.startswith("rel:")]
    metrics = ["k_rule"] + [c for c in df if re.search(r":(alpha_true|b_true|k_fit|rhoF_P|rhoF|cos_full|cos_diag)$", c)]
    corr = {}
    for tgt in targets:
        tgt_task = tgt.split(":", 1)[1] if tgt.startswith("rel:") else None
        for m in metrics:
            m_task = m.split(":", 1)[0] if ":" in m else None
            if tgt_task and m_task and m_task != tgt_task:
                continue  # per-task scores only against metrics measured on the same task
            corr[f"{tgt} ~ {m}"] = spearman_ci(df[m].to_numpy(float), df[tgt].to_numpy(float))

    strata = {}
    for mcol in MOTIF_COLS:
        if mcol not in df:
            continue
        has = df[df[mcol].astype(bool)]["objective"]
        not_ = df[~df[mcol].astype(bool)]["objective"]
        strata[mcol] = dict(n_with=int(len(has)), n_without=int(len(not_)),
                            median_with=float(has.median()) if len(has) else None,
                            median_without=float(not_.median()) if len(not_) else None)
    fams = {f: dict(n=int((df["family"] == f).sum()),
                    median=float(df.loc[df["family"] == f, "objective"].median()) if (df["family"] == f).any() else None)
            for f in FAMILIES}

    figs = []
    if scatter(df, "k_rule", "rule-level equivariance exponent k  (−1 SGD, 0 Adam/sign, +1 diag-NGD)",
               os.path.join(args.fig_dir, "pop_k_rule.png"), "Fitness vs equivariance exponent k"):
        figs.append("pop_k_rule.png")
    for col in [c for c in df if c.endswith(":rhoF_P")]:
        task = col.split(":")[0]
        name = f"pop_rhoF_P_{task}.png"
        if scatter(df, col, f"Fisher-cosine ρ_F of the rule's preconditioner on {task}  (1 = NGD)",
                   os.path.join(args.fig_dir, name), f"Fitness vs KL-efficiency of the preconditioner ({task})"):
            figs.append(name)

    out = dict(n_samples=int(len(df)), correlations=corr, motif_strata=strata, families=fams, figures=figs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    df.to_csv(os.path.splitext(args.out)[0] + "_table.csv", index=False)
    for k, v in corr.items():
        print(f"{k:<48} n={v['n']:<4} rho={v['rho']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print("families:", fams)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
