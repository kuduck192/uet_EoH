"""Build the report's tables and figures from the run and analysis outputs, and
fill the <!-- BEGIN:x --> ... <!-- END:x --> blocks of report/REPORT.md (the
prose around them is left untouched, so this can be re-run at any time).

    python make_report.py [--runs R1 R1_haiku ... R4] [--adir analysis_out] [--llm-adir analysis_out/haiku]

Two backbones side by side: the SLM (Qwen3.8-27B, runs R1-R4) and the LLM (Claude Haiku 5.5, runs
R1_haiku-R3_haiku). The LLM's rules are evaluated into merged copies of the SLM's files in --llm-adir
(scripts/validate_haiku.sh): the tables take the baselines and SLM rows from --adir and add the LLM rows.

Expected inputs in --adir (missing ones leave a placeholder in the report):
    eval_search.json, eval_transfer.json      evaluation/transfer.py (--merge-into adds rules)
    eval_r3.json, eval_r3_transfer.json       the same for the per-tensor rules (validation loss)
    alpha_<task>*.json                        analysis/alpha_fit.py --rules   (all files merged)
    invariance*.json                          analysis/invariance.py           (all files merged)
    ngd_alignment*.json                       analysis/ngd_alignment.py --rules (all files merged)
    population_corr.json                      analysis/population.py
    step_cost.json                            analysis/step_cost.py
    ablate_R1_1/{importance.md,minimal.py,trace.json}   analysis/ablate.py
    polish_<name>/{trace.json,<name>_polished.py}       analysis/polish.py
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "analysis"))

import plotstyle  # noqa: E402
from common import latest_population, load_eval_log, load_samples, load_seeds, run_dir  # noqa: E402
from harness import code_skeleton  # noqa: E402
from prob import aggregate_fitness  # noqa: E402

REPORT_DIR = os.path.join(HERE, "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
MISSING = "_(chưa có dữ liệu: chạy bước tương ứng trong `scripts/run_pipeline.sh`)_"
TASK_LABELS = {
    "softmax_illcond": "softmax ill-cond.", "mlp_fmnist": "MLP / F-MNIST", "cnn_cifar": "CNN / CIFAR-10",
    "charlm_shakespeare": "char-Transformer", "resnet18_cifar": "ResNet-18 / CIFAR-10",
    "gpt_shakespeare": "small GPT / Shakespeare", "mlp_fmnist_long": "MLP / F-MNIST, 10x steps",
    "cnn_cifar_bs512": "CNN / CIFAR-10, batch 512", "softmax_illcond_long": "softmax ill-cond., 10x steps",
    "tiny_fmnist": "tiny MLP / F-MNIST", "softmax_illcond_rot": "softmax ill-cond. (rotated)",
    # R3/R4 suites (validation loss, rule sees T)
    "softmax_rot_v": "softmax, rotated (T=150)", "mlp_fmnist_v": "MLP / F-MNIST (T=400)",
    "cnn_cifar_v": "CNN / CIFAR-10 (T=200)", "charlm_v": "char-Transformer (T=300)",
    "resnet18_cifar_v": "ResNet-18 / CIFAR-10 (T=2000)", "gpt_v": "small GPT / Shakespeare (T=2000)",
    "mlp_fmnist_long_v": "MLP / F-MNIST (T=3000)", "cnn_cifar_bs512_v": "CNN / CIFAR-10, batch 512 (T=1000)",
    "softmax_rot_long_v": "softmax, rotated (T=2000)",
}
RUNS = ["R1", "R1_haiku", "R2", "R2_haiku", "R3", "R3_haiku", "R4"]  # each SLM run next to its LLM twin
API_PRICE = {"claude-haiku-5-5": (0.10, 0.50)}  # USD per million input / output tokens, prompts up to 100K tokens
# Curves of one SLM and one LLM rule per panel, next to Adam and one baseline. Palette slots were checked
# pairwise per row with the data-viz palette validator; LLM rules are dash-dot as a second cue.
CURVE_ROWS = {
    "elementwise": ([["adam_ref", "R1#1", "R1_haiku#1"], ["adam_ref", "R2#1", "R2_haiku#1"]], ["R1", "R2"]),
    "tensor": ([["adam_ref", "shampoo_cosine", "R3#1", "R3_haiku_1_tolerant"],
                ["adam_ref", "muon_cosine", "R4#1", "R3_haiku_1_tolerant"]], ["R3 (blind)", "SLM-R4 (seeded)"]),
}
CURVE_SETTING = {"search": "elementwise", "transfer": "elementwise", "r3": "tensor", "r3_transfer": "tensor"}
SLOT = {"R1#1": 0, "R1_haiku#1": 2, "R2#1": 3, "R2_haiku#1": 7, "shampoo_cosine": 0, "R3#1": 4,
        "R3_haiku_1_tolerant": 2, "muon_cosine": 3, "R4#1": 6, "R1": 0, "R1_haiku": 2, "R2": 3, "R2_haiku": 7,
        "R3": 4, "R3_haiku": 2, "R4": 6}


def run_label(run):
    return f"LLM-{run[:-len('_haiku')]}" if run.endswith("_haiku") else f"SLM-{run}"


def _style(name, llm):
    """(colour, linestyle) of a rule or run in the backbone figures; Adam stays the neutral dashed reference."""
    if plotstyle._canon(name) == "adam":
        return plotstyle.INK2, "--"
    return plotstyle.SERIES[SLOT[name]], ("-." if llm else "-")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _f(x, spec="+.3f"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig_link(name, alt):
    return f"![{alt}](figures/{name})"


# ── EoH search ────────────────────────────────────────────────────────────────

def _robust_by_sample(run):
    """{sample_order: robust fitness} recomputed from each sample's per-task scores
    (a common yardstick for runs that optimised different objectives)."""
    rel = {cid: rec.get("relative") for cid, rec in load_eval_log(run).items()}
    out = {}
    for s in load_samples(run):
        r = rel.get(s.get("code_id"))
        if s.get("objective") is not None and r:
            out[s["sample_order"]] = aggregate_fitness(r, "robust")
    return out


def run_stats(run):
    samples = load_samples(run)
    if not samples:
        return None
    valid = [s for s in samples if s.get("objective") is not None]
    best = min(valid, key=lambda s: s["objective"])
    robust = _robust_by_sample(run)
    rb = min(robust, key=robust.get) if robust else None
    times = [s["time"] for s in samples if s.get("time")]
    cfg = _load(os.path.join(run_dir(run), "config.json")) or {}
    ops, ops_ok = Counter(s["operator"] for s in samples), Counter(s["operator"] for s in valid)
    pop = latest_population(run)
    seeds = {s["code"]: s["name"] for s in load_seeds(run)}
    best_pop = pop[0]
    return {
        "Backbone": (f"LLM: `{cfg['model']}` via the API, effort {cfg.get('effort')}, adaptive thinking"
                     if cfg.get("provider") == "anthropic" else
                     f"SLM: `{cfg.get('model', '?')}` on local vLLM, thinking {'on' if cfg.get('thinking') else 'off'}, "
                     f"T={cfg.get('temperature')}, top-p={cfg.get('top_p')}"),
        "Fitness optimised": ("robust (mean + worst-task excess)" if cfg.get("aggregate") == "robust"
                              else "mean over tasks")
                             + (", validation loss" if cfg.get("metric") == "val" else ", training loss"),
        "Rule interface": ("per tensor, `step(p, g, state, t, T, lr)`" if cfg.get("template") == "tensor"
                           else "element-wise, `step(p, g, state, t, lr)`"),
        "Initial population": _init_description(run, cfg, samples),
        "Population management": (f"pop {cfg.get('pop_size')}, max {cfg['skeleton_cap']} per code skeleton"
                                  if cfg.get("skeleton_cap") else f"pop {cfg.get('pop_size')}, EoH default"),
        "Samples (valid)": f"{len(samples)} ({len(valid)}, {100 * len(valid) / len(samples):.0f}%)",
        "Per operator (valid/total)": ", ".join(f"{op} {ops_ok[op]}/{ops[op]}" for op in sorted(ops)),
        "Best by own fitness": f"{best['objective']:+.4f} (#{best['sample_order']}, {best['operator']})",
        "Best in final population": f"{best_pop['objective']:+.4f}"
                                    + (f" (seed: {plotstyle.label(seeds[best_pop['code']])})"
                                       if best_pop["code"] in seeds else " (evolved)"),
        "Best robust fitness": f"{robust[rb]:+.4f} (#{rb})" if rb is not None else "–",
        "Beat Adam on all 4 tasks": str(sum(_all_better(run, o) for o in robust)),
        "Distinct skeletons in final population": f"{len({code_skeleton(i['code']) for i in pop})}/{len(pop)}",
        "Wall-clock": f"{(max(times) - min(times)) / 60:.0f} min" if len(times) > 1 else "–",
        "API cost (list price)": _api_cost(run),
    }


def _api_cost(run):
    """Calls, tokens and list-price cost of an API backbone, from the per-call log of api_anthropic.py."""
    path = os.path.join(run_dir(run), "results", "llm_usage.jsonl")
    if not os.path.exists(path):
        return "– (local GPU)"
    calls = tin = tout = 0
    usd = 0.0
    with open(path) as f:
        for line in f:
            r = json.loads(line) if line.strip() else {"error": "blank"}
            if "error" in r:
                continue
            pin, pout = API_PRICE[r["model"]]
            calls, tin, tout = calls + 1, tin + r["input_tokens"], tout + r["output_tokens"]
            usd += (r["input_tokens"] * pin + r["output_tokens"] * pout) / 1e6
    return f"${usd:.2f} ({calls} calls, {tin / 1e6:.1f}M input + {tout / 1e6:.1f}M output tokens)"


def _seed_population(run):
    """[(seed name, fitness)] of a seeded run's initial population, best first."""
    seeds = {s["code"]: s["name"] for s in load_seeds(run)}
    pop0 = _load(os.path.join(run_dir(run), "results", "pops", "population_generation_0.json")) or []
    return [(seeds[i["code"]], i["objective"]) for i in pop0 if i["code"] in seeds]


def _init_description(run, cfg, samples):
    if cfg.get("seeded"):
        return "seeds: " + ", ".join(f"{plotstyle.label(n)} {f:+.3f}" for n, f in _seed_population(run))
    n_i1 = sum(s["operator"] == "i1" for s in samples)
    return f"{n_i1} i1 samples from the SGD-momentum template"


_REL_CACHE = {}


def _all_better(run, order):
    if run not in _REL_CACHE:
        rel = {cid: rec.get("relative") for cid, rec in load_eval_log(run).items()}
        _REL_CACHE[run] = {s["sample_order"]: rel.get(s.get("code_id")) for s in load_samples(run)}
    r = _REL_CACHE[run].get(order)
    return bool(r) and max(r.values()) < 0


def eoh_summary(runs):
    stats = {r: run_stats(r) for r in runs}
    stats = {r: v for r, v in stats.items() if v}
    if not stats:
        return None
    keys = list(next(iter(stats.values())))
    return md_table([""] + [run_label(r) for r in stats], [[k] + [stats[r][k] for r in stats] for k in keys])


def fig_search_progress(runs):
    """Best-so-far fitness of each run next to its twin on the other backbone (LLM dash-dot), one panel
    per problem: R1, R2 (training loss, element-wise) and R3/R4 (validation loss, per tensor). A seeded
    run starts at its best seed (sample index 0); hand-written baselines are dashed grey lines."""
    import matplotlib.pyplot as plt

    data = {}
    for run in runs:
        samples = [s for s in load_samples(run) if s.get("objective") is not None]
        robust = _robust_by_sample(run)
        if not samples:
            continue
        rel = {cid: rec.get("relative") for cid, rec in load_eval_log(run).items()}
        x = np.array([s["sample_order"] for s in samples], float)
        mean = np.array([aggregate_fitness(rel[s["code_id"]], "mean") if rel.get(s["code_id"]) else np.nan
                         for s in samples])
        rob = np.array([robust.get(s["sample_order"], np.nan) for s in samples])
        seeds = _seed_population(run)
        if seeds:  # robust = own fitness for these runs
            best_seed = min(f for _, f in seeds)
            x, mean, rob = np.r_[0, x], np.r_[np.nan, mean], np.r_[best_seed, rob]
        data[run] = (x, mean, rob)
    if not data:
        return None
    panels = [(1, "R1: mean fitness, training loss", ("R1", "R1_haiku")),
              (2, "R2: robust fitness, training loss", ("R2", "R2_haiku")),
              (2, "R3/R4: robust fitness, validation loss", ("R3", "R3_haiku", "R4"))]
    panels = [(k, title, [r for r in rs if r in data]) for k, title, rs in panels]
    panels = [p for p in panels if p[2]]
    plotstyle.apply()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.6), squeeze=False)
    for ax, (k, title, rs) in zip(axes[0], panels):
        for run in rs:
            d = data[run]
            y = np.fmin.accumulate(np.where(np.isfinite(d[k]), d[k], np.inf))
            color, ls = _style(run, run.endswith("_haiku"))
            ax.step(d[0], y, where="post", lw=2, color=color, ls=ls,
                    label=run_label(run) + (" (seeded)" if _seed_population(run) else ""))
            ax.annotate(f"{y[-1]:+.3f}", (d[0][-1], y[-1]), xytext=(3, -10), textcoords="offset points",
                        fontsize=8, color=plotstyle.INK)
        if "R3" in rs:
            base = _load(os.path.join(HERE, "baselines_r3.json")) or {}
            for name in ("shampoo_cosine", "muon_cosine", "adamw_cosine"):
                if name in base:
                    ax.axhline(base[name]["robust"], color=plotstyle.OTHER, lw=1, ls="--")
                    ax.annotate(plotstyle.label(name), (0, base[name]["robust"]), xytext=(2, 2),
                                textcoords="offset points", fontsize=7, color=plotstyle.INK2)
        ax.axhline(0, color=plotstyle.INK2, lw=0.8)
        ax.set_title(f"best so far: {title}", loc="left", fontsize=9)
        ax.set_xlabel("sample index")
        ax.set_ylim(top=0.05)
        ax.legend(loc="upper right")
    axes[0][0].set_ylabel("fitness vs Adam (lower = better)")
    plotstyle.save(fig, os.path.join(FIG_DIR, "search_progress.png"))
    return fig_link("search_progress.png", "EoH search progress")


def top_rules(run, k=3):
    try:
        pop = latest_population(run)
    except FileNotFoundError:
        return None
    seeds = {s["code"]: s["name"] for s in load_seeds(run)}
    parts = []
    for i, ind in enumerate(pop[:k], start=1):
        seed = f" *(seed: {plotstyle.label(seeds[ind['code']])})*" if ind["code"] in seeds else ""
        parts.append(f"**{run_label(run)}#{i}** — fitness {ind['objective']:+.4f}{seed}\n\n"
                     f"> {ind.get('algorithm') or ''}\n\n"
                     f"```python\n{ind['code'].strip()}\n```")
    return "\n\n".join(parts)


def search_config(run):
    cfg = _load(os.path.join(run_dir(run), "config.json"))
    if not cfg:
        return None
    ref = cfg.get("reference", {})
    loss = "validation" if cfg.get("metric") == "val" else "training"
    return (f"**Task description sent to the LLM (verbatim):**\n\n> {cfg['task_description']}\n\n"
            f"**Template (initial body = SGD-momentum):**\n\n```python\n{cfg['template_program'].strip()}\n```\n\n"
            f"**Reference scores of LR-tuned Adam** (mean log {loss} loss over the whole run): "
            + ", ".join(f"{TASK_LABELS.get(t, t)} {v:+.3f}" for t, v in ref.items()))


# ── benchmark (evaluation/transfer.py) ───────────────────────────────────────

def benchmark_table(data):
    tbl = data["table"]
    tasks = list(next(iter(tbl.values())).keys())
    raised = data.get("raised", set())
    rows = []
    for name, row in tbl.items():
        cells = []
        for t in tasks:
            c = row[t]
            if "error" in c:
                cells.append("–" if c["error"] == "not run" else "error")
                continue
            sp = f"×{c['speedup']:.2f}" if c.get("speedup") else "never"
            div = f", {c['n_diverged']} diverged" if c.get("n_diverged") else ""
            mark = " †" if (name, t) in raised else ""
            cells.append(f"{c['rel_mean']:+.3f} ± {c['rel_std']:.3f} ({sp}{div}){mark}")
        rows.append([plotstyle.label(name)] + cells)
    note = ("\n\n† The first evaluation of this rule raised on this task (`torch.linalg.eigh` on a NaN or degenerate "
            "matrix); the value is from the rerun that scores an exception in `step()` as divergence from that step "
            "on (`scripts/validate_haiku.sh tolerant`)." if raised else "")
    return md_table(["optimizer"] + [TASK_LABELS.get(t, t) for t in tasks], rows) + note


def _raw_name(name):
    """R3_haiku_1_tolerant -> R3_haiku#1, the evaluation of the same rule without the exception wrapper."""
    m = re.match(r"^(R\d_haiku)_(\d+)_tolerant$", name)
    return f"{m.group(1)}#{m.group(2)}" if m else None


def _with_llm_rows(data, extra):
    """Add the LLM backbone's rules from the merged copy in --llm-adir (evaluated against the same Adam
    reference). A *_tolerant rerun replaces its raw row, and the cells where the raw run raised are kept
    in data["raised"]; baselines and SLM rules stay as in --adir."""
    if not extra:
        return data
    data = data or dict(table={}, raw=[], metric=extra.get("metric"))
    raised = data.setdefault("raised", set())
    rows = {n: r for n, r in extra["table"].items() if plotstyle.is_llm(n)}
    replaced = {_raw_name(n) for n in rows} - {None}
    tasks = list(next(iter((data["table"] or rows).values())).keys())
    for n, r in rows.items():
        if n in replaced:
            continue
        raw = rows.get(_raw_name(n), {})
        raised |= {(n, t) for t in tasks if raw.get(t, {}).get("error") not in (None, "not run")}
        data["table"][n] = {t: r.get(t, dict(error="not run")) for t in tasks}
    data["raw"] += [x for x in extra["raw"] if x["rule"] in rows and x["rule"] in data["table"]]
    return data


def _mean_curve(curves):
    arr = np.array([np.asarray(c, float) for c in curves])
    for row in arr:
        bad = ~np.isfinite(row)
        if bad.any():
            row[np.argmax(bad):] = np.nan
    return np.nanmean(arr, 0)


def _smooth(y, w=20):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def fig_loss_curves(data, name, title, rows, row_names):
    """One row per comparison (CURVE_ROWS), one column per task: the SLM and LLM rule of a row share each panel."""
    import matplotlib.pyplot as plt

    raw = {(r["rule"], r["task"]): r for r in data["raw"] if "curves" in r}
    tasks = list(next(iter(data["table"].values())).keys())
    val = data.get("metric") == "val"  # held-out loss every 10 steps instead of per-step minibatch loss
    w, every = (3, 10) if val else (20, 1)
    plotstyle.apply()
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=(2.75 * len(tasks), 2.35 * len(rows) + 0.55),
                             squeeze=False)
    for i, ents in enumerate(rows):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            for e in ents:
                if (e, t) not in raw:
                    continue
                y = _smooth(_mean_curve(raw[(e, t)]["curves"]), w)
                color, ls = _style(e, plotstyle.is_llm(e))
                ax.plot((np.arange(len(y)) + w) * every, y, color=color, ls=ls, lw=1.6, label=plotstyle.label(e))
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(TASK_LABELS.get(t, t), loc="left", fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("step")
        axes[i][0].set_ylabel(f"{row_names[i]}\n{'validation' if val else 'training'} loss")
    legend = {}  # from every panel: a rule may not have been run on the first task
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(lab, h)
    fig.legend(list(legend.values()), list(legend), loc="upper center", ncol=len(legend), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(rows) > 1 else 0.88))
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)
    return fig_link(name, title)


# ── NGD analyses ──────────────────────────────────────────────────────────────

ALIGN_TASKS = {"elementwise": ("tiny_fmnist", "softmax_illcond"), "tensor": ("tiny_fmnist", "softmax_illcond_rot")}


def _setting(name, v):
    """Which alignment table a rule belongs to: the per-tensor rules were measured on the rotated task."""
    if name == "full_ngd":
        return None  # reference row of both tables
    return "tensor" if "softmax_illcond_rot" in v.get("alignment", {}) else "elementwise"


def _moving_rows(a):
    """Checkpoints where the rule actually moved: at t = T a schedule that decays to zero gives a zero
    step, whose direction (and every cosine of it) is undefined."""
    return [r for r in a.get("rows", []) if r.get("kl", 0) > 0]


def _all_means(a):
    rows = _moving_rows(a)
    if not rows:
        return None
    keys = [k for k in rows[0] if k not in ("t", "loss")]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def alignment_table(data, setting="elementwise"):
    if not any(_setting(n, v) == setting for n, v in data.items()):
        return None  # only the full-NGD reference row
    data = {n: v for n, v in data.items() if _setting(n, v) in (setting, None)}
    tasks = [t for t in ALIGN_TASKS[setting] if any(t in v.get("alignment", {}) for v in data.values())]
    if not tasks:
        return None
    parts, refs = [], {t: [] for t in tasks}
    for t in tasks:
        rotation = t.startswith("softmax")
        headers = ["optimizer", "ρ_F(P)", "ρ_F(Δ)", "cos(Δ, full NGD)", "cos(Δ, diag-NGD)"]
        headers += ["rotation penalty"] if rotation else []
        rows = []
        for name, v in data.items():
            a = _all_means(v.get("alignment", {}).get(t) or {})
            row = [plotstyle.label(name)]
            if a is None:
                row += ["–"] * 4
            else:
                row += [_f(a["rhoF_P"], ".3f"), _f(a["rhoF"], "+.3f"), _f(a["cos_full"], "+.3f"),
                        _f(a["cos_diag"], "+.3f")]
                if not plotstyle.is_llm(name):  # the reference directions stay those of the SLM analysis
                    refs[t].append(a)
            if rotation:
                rot = v.get("rotation")
                row.append(_f(rot["penalty"], "+.3f") if rot else "–")
            rows.append(row)
        parts.append(f"**{TASK_LABELS[t]}**\n\n" + md_table(headers, rows))
    ref_rows = []
    for t in tasks:
        if refs[t]:
            m = {k: np.mean([a[k] for a in refs[t]]) for k in ("rhoF_sgd", "rhoF_adamF", "rhoF_diag", "rhoF_full", "diag_ref")}
            ref_rows.append([TASK_LABELS[t], f"{m['rhoF_sgd']:.3f}", f"{m['rhoF_adamF']:.3f}", f"{m['rhoF_diag']:.3f}",
                             f"{m['rhoF_full']:.3f}", f"{m['diag_ref']:+.3f}"])
    ref = md_table(["task", "ρ_F(−g) (SGD)", "ρ_F(−g/√diag F)", "ρ_F(−g/diag F) (diag-NGD)", "ρ_F(−F⁻¹g) (NGD)",
                    "cos(diag-NGD, NGD)"], ref_rows)
    return ("\n\n".join(parts) + "\n\n**Reference directions** (same probe gradient, averaged over the "
            "trajectories of the baselines and SLM rules above):\n\n" + ref)


def fig_alignment(data, setting="elementwise"):
    import matplotlib.pyplot as plt

    if not any(_setting(n, v) == setting for n, v in data.items()):
        return None
    data = {n: v for n, v in data.items() if _setting(n, v) in (setting, None)}
    tasks = [t for t in ALIGN_TASKS[setting] if any(t in v.get("alignment", {}) for v in data.values())]
    if not tasks:
        return None
    # rho_F(P) estimates a *diagonal* preconditioner: meaningless for matrix rules, so the per-tensor
    # setting plots the cosine between the actual step and the full-NGD direction instead
    metric, ylabel = (("rhoF_P", "ρ_F of the preconditioner") if setting == "elementwise"
                      else ("cos_full", "cos(step, full-NGD direction)"))
    plotstyle.apply()
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.8 * len(tasks), 3.1), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        for name, v in data.items():
            a = v.get("alignment", {}).get(t)
            rows = _moving_rows(a or {})
            if not rows or name.endswith(("#2", "#3")):  # the tables keep the runner-ups; the figure stays readable
                continue
            ax.plot([r["t"] for r in rows], [r[metric] for r in rows], color=plotstyle.color(name),
                    ls=plotstyle.linestyle(name), lw=1.6, label=plotstyle.label(name))
        ax.set_title(TASK_LABELS[t], loc="left")
        ax.set_xlabel("step")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_h = 0.28 * int(np.ceil(len(labels) / 4)) + 0.1  # inches above the panels, so the legend clears the titles
    fig.set_size_inches(3.8 * len(tasks), 2.7 + legend_h)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - legend_h / (2.7 + legend_h)))
    name = "alignment_rhoF.png" if setting == "elementwise" else f"alignment_rhoF_{setting}.png"
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)
    return fig_link(name, "Fisher-cosine along training")


def _alpha_summary(fit, source="true", warmup_frac=0.25):
    recs = [r for r in fit["records"] if r.get(source)]
    recs = recs[int(len(recs) * warmup_frac):] or recs
    if not recs:
        return None
    med = lambda k: float(np.median([r[source][k] for r in recs]))  # noqa: E731
    return dict(alpha=med("alpha"), b=med("b"), r2=med("r2"), k=2 * med("alpha") - med("b"))


def alpha_table(alphas, inv):
    names = []
    for d in alphas.values():
        names += [n for n in d if n not in names]
    if inv:
        names += [n for n in inv if n not in names]
    headers = ["optimizer"]
    for t in alphas:
        tag = TASK_LABELS[t]
        headers += [f"α ({tag})", f"b ({tag})", f"k = 2α−b ({tag})", f"R² ({tag})"]
    headers += ["k_rule (D=10)", "gap θ→φ, spread 10", "gap θ→φ, spread 100"]
    rows = []
    for n in names:
        row = [plotstyle.label(n)]
        for t, d in alphas.items():
            s = _alpha_summary(d[n]) if n in d else None
            row += ["–"] * 4 if s is None else [f"{s['alpha']:+.2f}", f"{s['b']:+.2f}", f"{s['k']:+.2f}", f"{s['r2']:.2f}"]
        if inv and n in inv:
            kr = next((r["median"] for r in inv[n]["rule_level"] if r["D"] == 10.0), None)
            gaps = {g["spread"]: g["gap"] for g in inv[n].get("training", [])}
            row += [_f(kr, "+.2f"), _f(gaps.get(10.0), "+.3f"), _f(gaps.get(100.0), "+.3f")]
        else:
            row += ["–"] * 3
        rows.append(row)
    return md_table(headers, rows)


def fig_alpha_plane(alphas):
    import matplotlib.pyplot as plt

    if not alphas:
        return None
    t = "softmax_illcond" if "softmax_illcond" in alphas else next(iter(alphas))
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for (b, a), lab in (((1, 0), "SGD"), ((1, 0.5), "Adam"), ((0, 0), "sign"), ((1, 1), "diag-NGD")):
        ax.scatter([b], [a], s=110, facecolor="none", edgecolor=plotstyle.INK2, linewidth=1.2, zorder=2)
        ax.annotate(f"theory: {lab}", (b, a), xytext=(8, 6), textcoords="offset points", fontsize=8,
                    color=plotstyle.INK2)
    for n, fit in alphas[t].items():
        s = _alpha_summary(fit)
        if s is None:
            continue
        evolved = n.startswith("R1") or n == "minimal"
        ax.scatter([s["b"]], [s["alpha"]], s=40, color=plotstyle.SERIES[0] if evolved else plotstyle.OTHER,
                   edgecolor=plotstyle.SURFACE, linewidth=1.2, zorder=3)
        ax.annotate(plotstyle.label(n).removeprefix("SLM-"), (s["b"], s["alpha"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=8, color=plotstyle.INK)  # SLM rules only (see main)
    ax.set_xlabel("b  (exponent of |m|)")
    ax.set_ylabel("α  (exponent of diag F)")
    ax.set_title(f"log|Δ| = c + b·log|m| − α·log F_ii  ({TASK_LABELS[t]})", loc="left", fontsize=9)
    ax.set_xlim(-0.3, 1.45)
    ax.set_ylim(-0.2, 1.25)
    plotstyle.save(fig, os.path.join(FIG_DIR, "alpha_plane.png"))
    return fig_link("alpha_plane.png", "(b, alpha) plane")


def population_table(data):
    rows = []
    for key, v in data["correlations"].items():
        tgt, metric = key.split(" ~ ")
        tgt = "fitness" if tgt == "objective" else f"task score: {TASK_LABELS.get(tgt[4:], tgt[4:])}"
        rows.append([tgt, f"`{metric}`", v["n"], _f(v["rho"], "+.2f"), f"[{_f(v['lo'], '+.2f')}, {_f(v['hi'], '+.2f')}]"])
    fam = md_table(["family (motif in code)", "samples", "median fitness"],
                   [[f, v["n"], _f(v["median"], "+.3f")] for f, v in data["families"].items()])
    figs = "\n\n".join(fig_link(f, f) for f in data.get("figures", []))
    return (md_table(["outcome", "NGD-likeness metric", "n", "Spearman ρ", "95% CI (bootstrap)"], rows)
            + "\n\n" + fam + ("\n\n" + figs if figs else ""))


def ablation_block(adir, run):
    d = os.path.join(adir, f"ablate_{run}_1")
    imp, mini, trace = (os.path.join(d, f) for f in ("importance.md", "minimal.py", "trace.json"))
    if not os.path.exists(imp):
        return None
    with open(imp) as f:
        out = f.read().split("\n", 1)[1].strip()  # drop the file's own title line
    tr = _load(trace)
    if tr:
        out += "\n\n**Greedy simplification trace:**\n\n" + md_table(
            ["round", "edit", "fitness"], [[r["round"], f"`{r['edit']}`", f"{r['fitness']:+.4f}"] for r in tr])
    if os.path.exists(mini):
        with open(mini) as f:
            out += f"\n\n**Minimal form:**\n\n```python\n{f.read().strip()}\n```"
    return out


GROUPS = [  # (name, motif tags from analysis/motifs.py; a sample is in the group if it has any of them)
    ("1. Momentum / averaging", ["momentum"]),
    ("2. Adaptive diagonal scaling (second moment, sign)", ["second_moment", "sign"]),
    ("3. Step-size schedule (t or t/T)", ["t_schedule", "uses_T", "cosine"]),
    ("4. Regularisation / robustness (weight decay, clipping)", ["uses_p", "saturation"]),
    ("5. Matrix (second-order) preconditioning", ["matrix_ops"]),
    ("5b. Layer-wise normalisation (per-tensor norms)", ["tensor_reduction"]),
]


def _truthy(v):
    return str(v).strip().lower() in ("true", "1")


def groups_table(runs):
    """Prevalence and best fitness of each mechanism group, per run."""
    import csv

    from motifs import tag

    cols, rows = [], {g: [] for g, _ in GROUPS}
    for run in runs:
        path = os.path.join(run_dir(run), "analysis", "samples.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            samples = [r for r in csv.DictReader(f) if r["objective"] not in ("", "None")]
        if not samples:
            continue
        pop_tags = [tag(ind["code"]) for ind in latest_population(run)]
        cols.append(run_label(run))
        for g, keys in GROUPS:
            has = [r for r in samples if any(_truthy(r.get(k)) for k in keys)]
            in_pop = sum(any(t.get(k) for k in keys) for t in pop_tags)
            best = min((float(r["objective"]) for r in has), default=None)
            rows[g].append(f"{100 * len(has) / len(samples):.0f}% / {in_pop}/{len(pop_tags)} / "
                           f"{_f(best, '+.3f')}")
    if not cols:
        return None
    return md_table(["mechanism group (% samples / in final pop / best fitness)"] + cols,
                    [[g] + rows[g] for g, _ in GROUPS])


def polish_block(adir):
    parts = []
    for d in sorted(glob.glob(os.path.join(adir, "polish_*"))):
        tr = _load(os.path.join(d, "trace.json"))
        if not tr:
            continue
        name = os.path.basename(d)[len("polish_"):]
        tasks = list(tr["trace"][0]["rel"])
        rows = [[r["phase"], f"`{r['edit'][:70]}`", f"{r['fitness']:+.4f}"] +
                [f"{r['rel'][t]:+.3f}" for t in tasks] + [f"{r['time'] / 60:.0f}"] for r in tr["trace"]]
        part = (f"**{name}** (objective: {tr['args']['objective']}, horizons ×{tr['args']['scales']}, "
                f"seeds {tr['args']['seeds']})\n\n"
                + md_table(["phase", "accepted change", "fitness"] + [TASK_LABELS.get(t, t) for t in tasks] + ["min"],
                           rows))
        files = glob.glob(os.path.join(d, "*_polished.py")) or glob.glob(os.path.join(d, "best.py"))
        if files:
            with open(files[0]) as f:
                part += f"\n\n```python\n{f.read().strip()}\n```"
        parts.append(part)
    return "\n\n".join(parts) or None


def step_cost_table(data):
    """Milliseconds per training step and the ratio to Adam (the first rule measured)."""
    tasks = list(next(iter(data.values())).keys())
    ref = data.get("adam_t") or next(iter(data.values()))
    rows = [[plotstyle.label(name)] + [f"{1000 * v[t]:.1f} (×{v[t] / ref[t]:.1f})" for t in tasks]
            for name, v in data.items()]
    return md_table(["optimizer"] + [TASK_LABELS.get(t, t) for t in tasks], rows)


def _deep_update(dst, src, depth):
    for k, v in src.items():
        if depth > 1 and isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v, depth - 1)
        else:
            dst[k] = v


def _merged(adir, pattern, depth=1):
    """Union of the rule -> result dicts of every JSON file matching pattern (later files win).
    depth > 1 merges nested dicts instead of replacing them, e.g. depth=3 for the alignment
    files, where the full-NGD reference rule is measured on different tasks in different files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(adir, pattern))):
        d = _load(path)
        if isinstance(d, dict):
            _deep_update(out, d, depth)
    return out or None


def _eval_tables(adir, suite):
    """eval_<suite>.json plus rows from eval_<suite>-*.json (rules evaluated separately, each file with
    its own Adam reference). Columns are the union of tasks; missing cells are marked "not run"."""
    paths = [os.path.join(adir, f"eval_{suite}.json")] + sorted(glob.glob(os.path.join(adir, f"eval_{suite}-*.json")))
    tables = [d for d in map(_load, paths) if d]
    if not tables:
        return None
    out = dict(table={}, raw=[], metric=tables[0].get("metric"))
    tasks = []
    for d in tables:
        for row in d["table"].values():
            tasks += [t for t in row if t not in tasks]
    for i, d in enumerate(tables):
        for name, row in d["table"].items():
            if i and name == "adam_ref":
                continue  # one Adam reference row: the first file's
            out["table"][name] = {t: row.get(t, dict(error="not run")) for t in tasks}
        out["raw"] += [r for r in d["raw"] if not (i and r["rule"] == "adam_ref")]
    for row in out["table"].values():
        for t in tasks:
            row.setdefault(t, dict(error="not run"))
    return out


# ── assembly ──────────────────────────────────────────────────────────────────

def fill(report_path, blocks):
    with open(report_path) as f:
        text = f.read()
    for key, content in blocks.items():
        begin, end = f"<!-- BEGIN:{key} -->", f"<!-- END:{key} -->"
        if begin not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{content if content else MISSING}\n{end}{tail}"
    with open(report_path, "w") as f:
        f.write(text)


def _with_llm(base, extra):
    """base plus the LLM rules of extra; the anchors that extra re-measured (SLM rules, baselines) are
    ignored, and so is a *_tolerant rerun whose raw twin is there."""
    out = dict(base or {})
    for n, v in (extra or {}).items():
        if plotstyle.is_llm(n) and _raw_name(n) not in (extra or {}):
            out.setdefault(n, v)
    return out or None


def _slm_only(d):
    return {n: v for n, v in (d or {}).items() if not plotstyle.is_llm(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--adir", default=os.path.join(HERE, "analysis_out"))
    ap.add_argument("--llm-adir", default=os.path.join(HERE, "analysis_out", "haiku"))
    ap.add_argument("--report", default=os.path.join(REPORT_DIR, "REPORT.md"))
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    A = lambda name: os.path.join(args.adir, name)  # noqa: E731
    runs = [r for r in args.runs if os.path.isdir(run_dir(r))]

    blocks = {"eoh_summary": eoh_summary(runs), "search_progress": fig_search_progress(runs)}
    for run in runs:
        if not run.endswith("_haiku"):  # the LLM twin got the same task description and template
            blocks[f"search_config_{run}"] = search_config(run)
        blocks[f"top_rules_{run}"] = top_rules(run)
        motifs_md = os.path.join(run_dir(run), "analysis", "timeline.md")
        if os.path.exists(motifs_md):
            with open(motifs_md) as f:
                blocks[f"timeline_{run}"] = f.read().split("\n", 1)[1].strip()

    for suite in ("search", "transfer", "r3", "r3_transfer"):
        data = _with_llm_rows(_eval_tables(args.adir, suite), _eval_tables(args.llm_adir, suite))
        blocks[f"bench_{suite}"] = benchmark_table(data) if data else None
        rows, row_names = CURVE_ROWS[CURVE_SETTING[suite]]
        blocks[f"curves_{suite}"] = fig_loss_curves(data, f"loss_{suite}.png", suite, rows, row_names) if data else None

    # the NGD tables add the LLM rules; the NGD figures stay those of the SLM analysis
    align = _with_llm(_merged(args.adir, "ngd_alignment*.json", depth=3),
                      _merged(args.llm_adir, "ngd_alignment*.json", depth=3))
    for setting, key in (("elementwise", "alignment"), ("tensor", "alignment_r3")):
        blocks[key] = alignment_table(align, setting) if align else None
        blocks[f"{key}_fig"] = fig_alignment(_slm_only(align), setting) if align else None

    alphas = {t: _with_llm(_merged(args.adir, f"alpha_{t}*.json"), _merged(args.llm_adir, f"alpha_{t}*.json"))
              for t in ("softmax_illcond", "mlp_fmnist")}
    alphas = {t: d for t, d in alphas.items() if d}
    inv = _with_llm(_merged(args.adir, "invariance*.json"), _merged(args.llm_adir, "invariance*.json"))
    blocks["alpha"] = alpha_table(alphas, inv) if (alphas or inv) else None
    blocks["alpha_fig"] = fig_alpha_plane({t: _slm_only(d) for t, d in alphas.items()})

    pop = _load(A("population_corr.json"))
    blocks["population"] = population_table(pop) if pop else None
    blocks["ablation"] = ablation_block(args.adir, "R1")
    blocks["polish"] = polish_block(args.adir)
    # one measurement of every rule of both backbones in a row, when validate_haiku.sh made one
    cost = _load(os.path.join(args.llm_adir, "step_cost.json")) or _load(A("step_cost.json"))
    blocks["step_cost"] = step_cost_table(cost) if cost else None
    blocks["groups"] = groups_table(runs)

    fill(args.report, blocks)
    done = [k for k, v in blocks.items() if v]
    print(f"filled {len(done)}/{len(blocks)} blocks: {', '.join(done)}")


if __name__ == "__main__":
    main()
