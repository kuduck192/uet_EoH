"""L0 - mine an EoH run: motif tags, rule-level equivariance, genealogy, thought timeline.

    python analysis/motifs.py R2            # writes runs/R2/analysis/{samples.csv,timeline.md,lineage.dot,*.png}

Motifs are detected on the source of step() (comments and docstrings removed).
k_rule is the rule-level equivariance exponent from invariance.py (D = 10):
-1 SGD-like, 0 Adam/sign-like, +1 diagonal-NGD-like.
"""

import argparse
import ast
import csv
import glob
import json
import os
import re

import numpy as np

from common import code_id, load_rule, load_samples, load_seeds, run_dir
from harness import check_elementwise, strip_docstrings, takes_T
from invariance import rule_exponent

MOTIFS = {
    # an exponential moving average or accumulation of the gradient (heavy ball, Adam's m, ...)
    "momentum": r"\.add_\(\s*g\b|\.lerp_\(\s*g\b|\.mul_\([^()]*\)\.add_\(\s*g|[\w.]+\s*\*\s*\w+\s*\+\s*\(?\s*1(\.0)?\s*-[^\n]*\*\s*g\b|\w+\s*\*\s*\w+\s*\+\s*g\b",
    "second_moment": r"\bg\s*\*\s*g\b|\bg\s*\*\*\s*2|square\(\s*g\b|addcmul_\(\s*g\s*,\s*g\b|\bg\.pow\(\s*2|\bg\.square\(",
    "fisher_moment": r"\bf\s*\*\s*f\b|\bf\s*\*\*\s*2|square\(\s*f\b|addcmul_\(\s*f\s*,\s*f\b|\bf\.pow\(\s*2|\bf\.square\(",
    "uses_f": r"\bf\b",
    "sqrt": r"sqrt|rsqrt|\*\*\s*\(?\s*0?\.5\b|pow\(\s*0?\.5\s*\)",
    "other_power": r"\*\*\s*\(?\s*-?\s*0?\.(?!5\b)\d+",
    "sign": r"\bsign\(",
    "saturation": r"tanh|clamp|clip|atan|sigmoid|softsign|\berf\(|minimum|maximum",
    "abs": r"\babs\(|\.abs\(",
    "bias_correction": r"\*\*\s*t\b|pow\([^()]*\bt\s*\)",
    "exp_log": r"\bexp\(|\blog\(|\blog1p\(",
    # the parameter *value* enters the update (weight decay, relative steps); not p.shape etc.
    "uses_p": r"(?<![\w.])p\b(?!\s*\.\s*(?:shape|dim|ndim|numel|size|device|dtype)\b)(?!\s*\))(?=\s*[*/+\-]|\s*\.(?:abs|pow|norm|sign)\b)|[*/+\-]\s*p\b(?!\s*\.)",
    # per-tensor rules (R3)
    # linalg norms are scalar reductions (layer-wise normalisation), not matrix preconditioning
    "matrix_ops": r"linalg\.(?!(?:vector_|matrix_)?norm\b)|\beigh\(|\bsvd\(|_lowrank\(|cholesky\(|\binverse\(|"
                  r"\bpinv\(|matmul\(|\s@\s|\.mm\(|\.T\b|\.t\(\)|\.transpose\(|\bkron\(",
    "tensor_reduction": r"\.norm\(|torch\.norm|linalg\.(?:vector_|matrix_)?norm\(|\.mean\(|\.sum\(|\.max\(\)|"
                        r"\.amax\(|\.std\(",
    "uses_T": r"\bT\b",
    "cosine": r"\bcos\(",
}


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.mapping.get(node.arg, node.arg)
        return node


def _canonical_args(step):
    """Rename step()'s positional parameters to the template names (p, g, [f,] state, t, lr)
    so the motif regexes still match when the LLM renames them (e.g. g -> grad)."""
    names = [a.arg for a in step.args.args]
    if len(names) > 3 and names[3] == "f":
        canon = ("self", "p", "g", "f", "state", "t", "lr")
    elif "T" in names:
        canon = ("self", "p", "g", "state", "t", "T", "lr")
    else:
        canon = ("self", "p", "g", "state", "t", "lr")
    used = {n.id for n in ast.walk(step) if isinstance(n, ast.Name)}
    mapping = {old: new for old, new in zip(names, canon) if old != new and new not in used}
    return _Rename(mapping).visit(step) if mapping else step


def method_sources(code):
    """{method name: source without comments/docstrings} for the UpdateRule class."""
    tree = strip_docstrings(ast.parse(code))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return {m.name: ast.unparse(_canonical_args(m) if m.name == "step" else m)
                    for m in node.body if isinstance(m, ast.FunctionDef)}
    return {}


def tag(code):
    try:
        methods = method_sources(code)
        tree = strip_docstrings(ast.parse(code))
        helpers = [ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:  # failed samples can hold unparsable LLM output
        methods, helpers = {}, []
    step = methods.get("step", "")
    body = step.split("\n", 1)[1] if "\n" in step else step  # drop the signature line
    # helper methods and module-level functions called by step() count as part of it
    body = "\n".join([body] + [src for name, src in methods.items() if name not in ("step", "init_state", "__init__")]
                      + helpers)
    tags = {name: bool(re.search(rx, body)) for name, rx in MOTIFS.items()}
    # Schedules: t used other than as a bias-correction exponent.
    t_uses = len(re.findall(r"\bt\b", body))
    bc_uses = len(re.findall(r"\*\*\s*t\b|pow\([^()]*\bt\s*\)", body))
    tags["t_schedule"] = t_uses > bc_uses
    tags["n_state"] = len(re.findall(r"zeros_like|ones_like|full_like|clone\(", methods.get("init_state", "")))
    tags["n_lines"] = body.count("\n") + 1
    return tags


def _plot(rows, pops, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = [r for r in rows if r["objective"] is not None]
    if not valid:
        return
    x = np.array([r["sample_order"] for r in valid])
    y = np.array([r["objective"] for r in valid])
    best = np.minimum.accumulate(y)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ops = sorted({r["operator"] for r in valid})
    for op in ops:
        sel = np.array([r["operator"] == op for r in valid])
        ax.scatter(x[sel], y[sel], s=8, alpha=0.5, label=op)
    ax.plot(x, best, color="k", lw=1.5, label="best so far")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("sample")
    ax.set_ylabel("fitness (vs Adam, lower is better)")
    ax.set_ylim(min(y.min(), 0) - 0.05, np.percentile(y, 90) + 0.05)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fitness.png"), dpi=150)
    plt.close(fig)

    kv = [r for r in valid if np.isfinite(r["k_rule"])]
    if kv:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sc = ax.scatter([r["sample_order"] for r in kv], [r["k_rule"] for r in kv],
                        c=[r["objective"] for r in kv], cmap="viridis_r", s=10,
                        vmax=np.percentile([r["objective"] for r in kv], 90))
        for k, label in ((-1, "SGD"), (0, "Adam / sign"), (1, "diag NGD")):
            ax.axhline(k, color="gray", lw=0.8, ls="--")
            ax.text(x.max(), k, f" {label}", va="center", fontsize=7)
        ax.set_xlabel("sample")
        ax.set_ylabel("equivariance exponent k")
        fig.colorbar(sc, label="fitness")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "k_rule.png"), dpi=150)
        plt.close(fig)

    if pops:
        gens = sorted(pops)
        names = [m for m in MOTIFS if any(pops[g_][m] for g_ in gens)] + ["t_schedule"]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for m in names:
            ax.plot(gens, [pops[g_][m] for g_ in gens], label=m)
        ax.set_xlabel("generation (checkpoint)")
        ax.set_ylabel("fraction of population")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "motifs.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--no-k", action="store_true", help="skip the rule-level exponent (no GPU needed)")
    args = ap.parse_args()

    out_dir = os.path.join(run_dir(args.run), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    known_k = {}  # k_rule of an earlier pass over this run (it is the slow part)
    prev = os.path.join(out_dir, "samples.csv")
    if os.path.exists(prev):
        with open(prev) as f:
            for row in csv.DictReader(f):
                if row.get("k_rule") not in (None, "", "nan"):
                    known_k[row["code_id"]] = float(row["k_rule"])
    samples = load_samples(args.run)
    seeds = load_seeds(args.run)

    nodes = {}  # code_id -> record, for the genealogy
    rows = []
    best = None
    for i, s in enumerate(seeds):
        cid = code_id(s["code"])
        nodes[cid] = dict(name=f"seed{i}", algorithm=s["algorithm"], parents=[], objective=None)
    for s in samples:
        if not s.get("code"):
            continue
        r = dict(sample_order=s["sample_order"], operator=s["operator"], objective=s["objective"],
                 code_id=s["code_id"], parents=" ".join(s.get("parents") or []),
                 algorithm=(s.get("algorithm") or "").replace("\n", " "))
        r.update(tag(s["code"]))
        r["k_rule"] = known_k.get(s["code_id"], float("nan"))
        if not args.no_k and s["objective"] is not None and s["code_id"] not in known_k:
            try:
                cls, fisher = load_rule(s)
                if not takes_T(cls) and check_elementwise(cls, fisher) is None:
                    r["k_rule"] = rule_exponent(cls, fisher, 10.0, n=4096, T=30)["median"]
            except Exception:
                pass
        if s["objective"] is not None:
            best = s["objective"] if best is None else min(best, s["objective"])
        r["best_so_far"] = best
        rows.append(r)
        nodes.setdefault(s["code_id"], dict(name=f"#{s['sample_order']}", algorithm=r["algorithm"],
                                            parents=s.get("parents") or [], objective=s["objective"]))

    if not rows:
        print(f"no samples in {run_dir(args.run)}")
        return
    with open(os.path.join(out_dir, "samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Motif prevalence in each checkpointed population.
    pops = {}
    for path in glob.glob(os.path.join(run_dir(args.run), "results", "pops", "population_generation_*.json")):
        gen = int(re.search(r"_(\d+)\.json$", path).group(1))
        with open(path) as f:
            pop = json.load(f)
        tags = [tag(ind["code"]) for ind in pop]
        pops[gen] = {m: float(np.mean([t[m] for t in tags])) for m in list(MOTIFS) + ["t_schedule"]}
    with open(os.path.join(out_dir, "population_motifs.json"), "w") as f:
        json.dump(pops, f, indent=2)

    # Timeline of improvements: thought + motif changes at each new best.
    lines, prev, prev_tags = ["# New-best timeline\n"], None, None
    for r in rows:
        if r["objective"] is None or (prev is not None and r["objective"] >= prev):
            continue
        cur_tags = {m: r[m] for m in list(MOTIFS) + ["t_schedule"]}
        diff = ""
        if prev_tags:
            gained = [m for m in cur_tags if cur_tags[m] and not prev_tags[m]]
            lost = [m for m in cur_tags if prev_tags[m] and not cur_tags[m]]
            diff = " ".join([f"+{m}" for m in gained] + [f"-{m}" for m in lost])
        k = f"k={r['k_rule']:+.2f} " if r["k_rule"] == r["k_rule"] else ""  # NaN for per-tensor rules
        lines.append(f"- **#{r['sample_order']}** [{r['operator']}] fitness {r['objective']:+.4f} "
                     f"{k}{diff}\n  > {r['algorithm']}\n")
        prev, prev_tags = r["objective"], cur_tags
    with open(os.path.join(out_dir, "timeline.md"), "w") as f:
        f.write("\n".join(lines))

    # Lineage of the best sample as a Graphviz DOT file.
    valid = [r for r in rows if r["objective"] is not None]
    if valid:
        top = min(valid, key=lambda r: r["objective"])["code_id"]
        edges, seen, stack = [], set(), [top]
        while stack:
            cid = stack.pop()
            if cid in seen or cid not in nodes:
                continue
            seen.add(cid)
            for parent in nodes[cid]["parents"]:
                edges.append((parent, cid))
                stack.append(parent)
        with open(os.path.join(out_dir, "lineage.dot"), "w") as f:
            f.write("digraph lineage {\n  rankdir=LR; node [shape=box, fontsize=9];\n")
            for cid in seen:
                n = nodes[cid]
                obj = "" if n["objective"] is None else f"\\n{n['objective']:+.4f}"
                label = (n["algorithm"] or "")[:70].replace('"', "'")
                f.write(f'  "{cid}" [label="{n["name"]}{obj}\\n{label}"];\n')
            for a, b in edges:
                if a in seen:
                    f.write(f'  "{a}" -> "{b}";\n')
            f.write("}\n")

    _plot(rows, pops, out_dir)
    print(f"{len(rows)} samples, best {best}; wrote {out_dir}")


if __name__ == "__main__":
    main()
