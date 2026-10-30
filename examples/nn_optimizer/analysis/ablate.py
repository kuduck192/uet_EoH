"""L1 - greedy ablation of an evolved rule down to a minimal canonical form.

Candidate edits inside step(): drop a statement, unwrap a call f(x, ...) -> x,
keep one side of a binary operation, keep one branch of a conditional
expression. Round 0 scores every edit (the component-importance table); then
the best edit whose fitness is within --tol of the current rule is applied,
repeatedly, until no edit is acceptable.

    python analysis/ablate.py --rule run:R2:1 --workers 8 [--steps-scale 0.5]

Writes analysis_out/ablate_<name>/{importance.md,trace.json,minimal.py}.
"""

import argparse
import ast
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from common import ROOT, load_rule, resolve_rules
from harness import check_rule, compile_rule, strip_docstrings
from prob import reference_scores, score_rule
from tasks import SEARCH_SUITE


def _step_nodes(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == "step":
                    return [n for n in ast.walk(m) if n is not m]
    return []


def _parents(tree):
    out = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for i, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        out[id(child)] = (parent, field, i)
            elif isinstance(value, ast.AST):
                out[id(value)] = (parent, field, None)
    return out


def _apply(code, idx, kind):
    tree = ast.parse(code)
    target = _step_nodes(tree)[idx]
    parent, field, i = _parents(tree)[id(target)]
    if kind == "drop":
        body = getattr(parent, field)
        del body[i]
        if not body:
            body.append(ast.Pass())
    else:
        new = {"unwrap": lambda n: n.args[0], "left": lambda n: n.left, "right": lambda n: n.right,
               "body": lambda n: n.body, "orelse": lambda n: n.orelse}[kind](target)
        if i is None:
            setattr(parent, field, new)
        else:
            getattr(parent, field)[i] = new
    return ast.unparse(ast.fix_missing_locations(tree))


def candidate_edits(code):
    """[(description, new_code)] with duplicates and no-ops removed."""
    tree = ast.parse(code)
    out, seen = [], {ast.unparse(tree)}
    for idx, node in enumerate(_step_nodes(tree)):
        kinds = []
        if isinstance(node, ast.stmt) and not isinstance(node, ast.Return):
            kinds.append("drop")
        if isinstance(node, ast.Call) and node.args:
            kinds.append("unwrap")
        if isinstance(node, ast.BinOp):
            kinds += ["left", "right"]
        if isinstance(node, ast.IfExp):
            kinds += ["body", "orelse"]
        for kind in kinds:
            try:
                new = _apply(code, idx, kind)
                ast.parse(new)
            except Exception:
                continue
            if new in seen:
                continue
            seen.add(new)
            out.append((f"{kind}: {ast.unparse(node)[:90]}", new))
    return out


def _fitness(code, tasks, use_fisher, steps_scale, reference):
    try:
        cls = compile_rule(code)
        if check_rule(cls, use_fisher) is not None:
            return None
        res = score_rule(cls, tasks, use_fisher, steps_scale=steps_scale)
        return sum(r["score"] - reference[n] for n, r in res.items()) / len(res)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--tasks", nargs="+", default=SEARCH_SUITE)
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.01, help="max fitness loss accepted per edit")
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rule = resolve_rules([args.rule])[0]
    _, use_fisher = load_rule(rule)
    reference = reference_scores(args.tasks, steps_scale=args.steps_scale)
    out_dir = os.path.join(ROOT, "analysis_out", f"ablate_{rule['name'].replace('#', '_')}")
    os.makedirs(out_dir, exist_ok=True)

    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))

    def score_all(codes):
        futs = [pool.submit(_fitness, c, args.tasks, use_fisher, args.steps_scale, reference) for c in codes]
        return [f.result() for f in futs]

    # Docstrings are statements too; strip them so no evaluation is spent on removing one.
    code = ast.unparse(strip_docstrings(ast.parse(rule["code"])))
    fit = score_all([code])[0]
    print(f"base fitness {fit:+.4f}", flush=True)
    trace = [dict(round=0, edit="original", fitness=fit, code=code)]

    for rnd in range(1, args.max_rounds + 1):
        edits = candidate_edits(code)
        scores = score_all([c for _, c in edits])
        scored = [(d, c, s) for (d, c), s in zip(edits, scores)]
        if rnd == 1:
            with open(os.path.join(out_dir, "importance.md"), "w") as f:
                f.write(f"# Component importance for {rule['name']} (base fitness {fit:+.4f})\n\n"
                        "| edit | fitness | change |\n|---|---|---|\n")
                # most harmful removals first; edits that break the rule last
                for d, _, s in sorted(scored, key=lambda e: (e[2] is None, -(e[2] or 0.0))):
                    change = "invalid" if s is None else f"{s - fit:+.4f}"
                    f.write(f"| `{d}` | {'-' if s is None else f'{s:+.4f}'} | {change} |\n")
        ok = [(d, c, s) for d, c, s in scored if s is not None and s <= fit + args.tol]
        if not ok:
            break
        d, code, new_fit = min(ok, key=lambda e: (e[2], len(e[1])))
        print(f"round {rnd}: {d}  ->  {new_fit:+.4f} ({new_fit - fit:+.4f})", flush=True)
        fit = new_fit
        trace.append(dict(round=rnd, edit=d, fitness=fit, code=code))

    pool.shutdown()
    with open(os.path.join(out_dir, "trace.json"), "w") as f:
        json.dump(trace, f, indent=2)
    with open(os.path.join(out_dir, "minimal.py"), "w") as f:
        f.write(f"# minimal form of {rule['name']}: fitness {fit:+.4f} "
                f"(original {trace[0]['fitness']:+.4f})\n{code}\n")
    print(f"minimal fitness {fit:+.4f}; wrote {out_dir}")


if __name__ == "__main__":
    main()
