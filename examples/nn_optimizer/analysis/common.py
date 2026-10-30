"""Shared helpers for the analysis scripts: locating runs and resolving rules.

Rule specs accepted on every command line:
    baseline:<name>      a hand-designed rule from baselines.py (e.g. baseline:adam)
    run:<run>[:<k>]      the k best rules of the latest population of runs/<run> (default k=1)
    sample:<run>:<n>     sample number n of runs/<run> (named <run>@<n>)
    file:<path>          a Python file defining UpdateRule
"""

import glob
import hashlib
import inspect
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '..', '..', 'eoh', 'src'))

from baselines import BASELINES, FISHER_BASELINES, TENSOR_BASELINES  # noqa: E402
from harness import compile_rule  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "runs")


def code_id(code):
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]


def run_dir(run):
    return run if os.path.isdir(run) else os.path.join(RUNS_DIR, run)


def latest_population(run):
    """Latest checkpointed population of a run, best first."""
    files = glob.glob(os.path.join(run_dir(run), "results", "pops", "population_generation_*.json"))
    if not files:
        raise FileNotFoundError(f"no populations under {run_dir(run)}")
    latest = max(files, key=lambda p: int(re.search(r"_(\d+)\.json$", p).group(1)))
    with open(latest) as f:
        pop = json.load(f)
    return sorted(pop, key=lambda ind: ind["objective"])


def load_samples(run):
    """All evaluated samples of a run in generation order (excludes seeds)."""
    out = []
    for path in glob.glob(os.path.join(run_dir(run), "results", "samples", "samples_*~*.json")):
        with open(path) as f:
            out.extend(json.load(f))
    out.sort(key=lambda s: s["sample_order"])
    for s in out:
        if s.get("code") and not s.get("code_id"):
            s["code_id"] = code_id(s["code"])
    return out


def valid_samples(run, subsample=None, seed=0):
    """Valid samples of a run; with subsample=n a fixed random subset of n of them, so that
    every --all-samples analysis (alpha_fit, ngd_alignment) sees the same samples."""
    out = [s for s in load_samples(run) if s.get("objective") is not None and s.get("code")]
    if subsample and subsample < len(out):
        keep = set(random.Random(seed).sample([s["sample_order"] for s in out], subsample))
        out = [s for s in out if s["sample_order"] in keep]
    return out


def load_seeds(run):
    path = os.path.join(run_dir(run), "seeds.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_eval_log(run):
    """{code_id: latest evaluation record} from results/eval_log/*.json (one file per
    evaluation, written by prob.py), plus a legacy eval_log.jsonl if present."""
    res = os.path.join(run_dir(run), "results")
    records = []
    for path in glob.glob(os.path.join(res, "eval_log", "*.json")):
        try:
            with open(path) as f:
                records.append(json.load(f))
        except (OSError, ValueError):
            pass
    legacy = os.path.join(res, "eval_log.jsonl")
    if os.path.exists(legacy):
        with open(legacy) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
    out = {}
    for rec in sorted(records, key=lambda r: r.get("time", 0.0)):
        out[rec["code_id"]] = rec
    return out


def resolve_rules(specs):
    """Expand rule specs into a list of dicts(name, code, algorithm)."""
    rules = []
    for spec in specs:
        kind, _, rest = spec.partition(":")
        if kind == "baseline":
            b = {**BASELINES, **FISHER_BASELINES, **TENSOR_BASELINES}[rest]
            rules.append(dict(name=rest, code=b["code"], algorithm=b["algorithm"]))
        elif kind == "run":
            run, _, k = rest.partition(":")
            for i, ind in enumerate(latest_population(run)[: int(k or 1)]):
                rules.append(dict(name=f"{os.path.basename(run)}#{i + 1}", code=ind["code"],
                                  algorithm=ind.get("algorithm"), objective=ind["objective"]))
        elif kind == "sample":
            run, _, n = rest.partition(":")
            s = next(s for s in load_samples(run) if s["sample_order"] == int(n))
            rules.append(dict(name=f"{os.path.basename(run)}@{n}", code=s["code"],
                              algorithm=s.get("algorithm"), objective=s.get("objective")))
        elif kind == "file":
            with open(rest) as f:
                rules.append(dict(name=os.path.splitext(os.path.basename(rest))[0], code=f.read(),
                                  algorithm=None))
        else:
            raise ValueError(f"unknown rule spec {spec!r}")
    return rules


def uses_fisher(rule_cls):
    """True if step() takes the Fisher sample f (signature p, g, f, state, t, lr).

    Decided by the name of the fourth parameter (after self, p, g), not by the
    parameter count: evolved rules often append hyper-parameters with defaults.
    """
    params = list(inspect.signature(rule_cls.step).parameters)
    return len(params) > 3 and params[3] == "f"


def load_rule(rule):
    cls = compile_rule(rule["code"])
    return cls, uses_fisher(cls)
