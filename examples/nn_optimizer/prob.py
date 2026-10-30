"""EoH task: discover element-wise neural-network optimizers that converge fast.

The LLM designs an `UpdateRule` class. The harness applies it to the flattened
parameters of four small training problems, tunes the learning rate per task
(wide log probe + local refinement), and scores the whole training-loss curve.

Fitness (lower is better) = mean over tasks of
    mean_t log L_t(candidate)  -  mean_t log L_t(Adam)
with both sides LR-tuned the same way, so 0 means "as fast as Adam" and
negative values beat it.

    python prob.py --calibrate                 # compute / cache the Adam reference
    python prob.py --baselines [--fisher]      # score every hand-designed baseline
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from baselines import ADAM, baselines_for
from harness import (PROBE_FRAC, PROBE_LRS, REFINE, check_rule, check_speed,
                     check_tensor_speed, compile_rule, default_device, takes_T, tune_lr)
from tasks import R3_SUITE, SEARCH_SUITE, clear_cache, make_task

HERE = os.path.dirname(os.path.abspath(__file__))
REF_CACHE = os.path.join(HERE, "ref_cache.json")

_TEMPLATE = '''import torch
import math


class UpdateRule:
    """Element-wise optimizer, constructed with no arguments.

    One instance handles all parameters of the network, flattened into a single
    1-D tensor, with a single state dict. The update of each entry may depend
    only on that entry's own values (p, g{f_list}, state tensors) and on the
    scalars t and lr.
    """

    def init_state(self, p: torch.Tensor) -> dict:
        """Return the initial state: any number of tensors shaped like p."""
        return {state_init}

    def step(self, p: torch.Tensor, g: torch.Tensor,{f_param} state: dict, t: int, lr: float) -> torch.Tensor:
        """Return the update delta (same shape as p); the harness applies p += delta.

        Args:
            p:     current parameter values.
            g:     minibatch gradient of the training loss.{f_doc}
            state: the dict returned by init_state; update its tensors in place.
            t:     step counter, starting at 1.
            lr:    learning rate selected by the harness with a log-spaced sweep.
        """
{body}'''

_BODIES = {
    "sgd": ('{"m": torch.zeros_like(p)}',
            '        m = state["m"]\n'
            '        m.mul_(0.9).add_(g)\n'
            '        return -lr * m\n'),
    "adam": ('{"m": torch.zeros_like(p), "v": torch.zeros_like(p)}',
             '        m, v = state["m"], state["v"]\n'
             '        m.mul_(0.9).add_(g, alpha=0.1)\n'
             '        v.mul_(0.999).addcmul_(g, g, value=0.001)\n'
             '        m_hat = m / (1 - 0.9 ** t)\n'
             '        v_hat = v / (1 - 0.999 ** t)\n'
             '        return -lr * m_hat / (v_hat.sqrt() + 1e-8)\n'),
}

_F_DOC = ("\n            f:     Fisher sample: gradient of the loss with labels sampled from the model's own"
          "\n                   predictive distribution, scaled so that f * f is an unbiased estimate of the"
          "\n                   diagonal of the Fisher information matrix.")


_TENSOR_TEMPLATE = '''import torch
import math


class UpdateRule:
    """Optimizer for one parameter tensor; the harness builds one instance (with no
    arguments) and one state for every parameter tensor of the network.

    Tensors keep their natural shapes: weight matrices (out, in), convolution
    kernels (out, in, kh, kw), embeddings (vocab, dim), biases and normalisation
    gains (n,).
    """

    def init_state(self, p: torch.Tensor) -> dict:
        """Return the initial state of this tensor: tensors and numbers."""
        return {"m": torch.zeros_like(p)}

    def step(self, p: torch.Tensor, g: torch.Tensor, state: dict, t: int, T: int, lr: float) -> torch.Tensor:
        """Return the update of this tensor (same shape as p); the harness applies p += delta.

        Args:
            p:     current values of this parameter tensor.
            g:     minibatch gradient of the training loss with respect to p.
            state: the dict returned by init_state for this tensor; update it in place.
            t:     step counter, from 1 to T.
            T:     total number of training steps of this run.
            lr:    base learning rate, selected per task by the harness with a log-spaced sweep.
        """
        m = state["m"]
        m.mul_(0.9).add_(g)
        return -lr * m
'''


def make_tensor_description(aggregate="robust"):
    text = (
        "Design a novel optimizer (update rule) for training neural networks that decreases the "
        "held-out validation loss as fast as possible. The rule is applied separately to every "
        "parameter tensor (weight matrices, convolution kernels, embeddings, biases, normalisation "
        "gains) at every step; it receives the step counter t and the total number of steps T of the "
        "run, and the harness tunes the base learning rate lr per task. Fitness is the average log "
        "validation loss over the whole run (so early progress matters as much as the final value), "
        "relative to Adam with a tuned constant learning rate, on four training runs of different "
        "lengths T (200 to 600 steps): an ill-conditioned softmax regression with correlated input "
        "features, an MLP on Fashion-MNIST, a small CNN on CIFAR-10 and a small character-level "
        "transformer."
    )
    if aggregate == "robust":
        text += (
            " The fitness also adds the largest per-task excess over Adam, so a rule that is slower "
            "than Adam on any single task is penalised: aim to be at least as fast as Adam on every task."
        )
    text += (
        " The update may use any deterministic torch operations on its tensor: element-wise "
        "operations, reductions over the tensor (norms, means), and for matrices (reshape higher-order "
        "kernels to (out, -1)) matrix products and torch.linalg functions (e.g. eigh, solve, matrix "
        "powers); recompute expensive matrix functions only every few steps so that a step costs at "
        "most a few times an Adam step. Randomness is not allowed."
    )
    return text


def make_template(use_fisher, body="adam"):
    if body == "tensor":
        return _TENSOR_TEMPLATE
    state_init, code = _BODIES[body]
    return (_TEMPLATE
            .replace("{f_list}", ", f" if use_fisher else "")
            .replace("{f_param}", " f: torch.Tensor," if use_fisher else "")
            .replace("{f_doc}", _F_DOC if use_fisher else "")
            .replace("{state_init}", state_init)
            .replace("{body}", code))


def make_description(use_fisher, aggregate="mean"):
    text = (
        "Design a novel element-wise update rule (optimizer) for training neural networks that "
        "decreases the training loss as fast as possible. The rule is applied to all parameters "
        "at every step, and the harness tunes the learning rate lr separately for each task. "
        "Fitness is the average log training loss over the whole run (so early progress matters "
        "as much as the final loss), relative to Adam, averaged over four short training runs of a "
        "few hundred steps: ill-conditioned softmax regression, an MLP on Fashion-MNIST, a small "
        "CNN on CIFAR-10 and a small character-level transformer. The update of every coordinate "
        "may depend only on that coordinate's parameter value, gradient and its own state tensors, "
        "plus the step counter t and lr: any element-wise torch operation is allowed, but sums, "
        "means, norms, max or other reductions over elements, indexing, and use of the tensor shape "
        "or size are forbidden and are rejected automatically."
    )
    if aggregate == "robust":
        text += (
            " The fitness also adds the largest per-task excess over Adam, so a rule that is slower "
            "than Adam on any single task is penalised: aim to be at least as fast as Adam on every task."
        )
    if use_fisher:
        text += (
            " Besides the gradient g, the rule receives f, a Fisher sample: the gradient of the loss "
            "with labels sampled from the model's own predictive distribution, scaled so that f * f "
            "is an unbiased estimate of the diagonal Fisher information."
        )
    return text


def aggregate_fitness(rel, how="mean"):
    """Fitness from per-task scores relative to Adam: 'mean' averages them; 'robust'
    adds the worst task's excess over Adam, max(0, max_k rel_k)."""
    vals = list(rel.values())
    fitness = float(np.mean(vals))
    if how == "robust":
        fitness += max(0.0, max(vals))
    return fitness


def code_id(code):
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]


def _device_name():
    """GPU model name for the reference cache key. Read from the driver's /proc
    entry when possible so the EoH main process never creates a CUDA context
    (memory on shared GPUs is tight)."""
    import glob as _glob
    for path in sorted(_glob.glob("/proc/driver/nvidia/gpus/*/information")):
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("Model:"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    import torch
    dev = default_device()
    return torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"


# Cheaper lr search used by run R3 (rules there cost more per step): 4 probes, same refinement.
PROBE_LRS_R3 = (1e-4, 1e-3, 1e-2, 1e-1)


def score_rule(rule_cls, task_names, use_fisher, seed=0, steps_scale=1.0, device=None, metric="train",
               probe_lrs=PROBE_LRS):
    """Run tune_lr on every task. Returns {task_name: tune_lr result}."""
    device = device or default_device()
    out = {}
    for name in task_names:
        task = make_task(name)
        task.steps = max(20, int(round(task.steps * steps_scale)))
        out[name] = tune_lr(task, rule_cls, seed=seed, use_fisher=use_fisher, device=device, metric=metric,
                            probe_lrs=probe_lrs)
    return out


def reference_scores(task_names, seed=0, steps_scale=1.0, cache_path=REF_CACHE, verbose=True, compute=True,
                     metric="train", probe_lrs=PROBE_LRS):
    """LR-tuned Adam curve scores per task, cached on disk per configuration and device.

    With compute=False a cache miss is an error instead of a training run (the
    EoH parent process must not create a CUDA context).
    """
    key = dict(tasks=sorted(task_names), seed=seed, steps_scale=steps_scale, probe=list(probe_lrs),
               frac=PROBE_FRAC, refine=list(REFINE), device=_device_name())
    if metric != "train":  # keeps the keys of training-loss references unchanged
        key["metric"] = metric
        key["steps"] = {n: make_task(n).steps for n in sorted(task_names)}
    key = json.dumps(key, sort_keys=True)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    if key not in cache:
        if not compute:
            raise RuntimeError(
                "no cached Adam reference for this configuration; run "
                f"`python prob.py --calibrate --tasks {' '.join(task_names)} --steps-scale {steps_scale}` first")
        if verbose:
            print(f"[reference] computing Adam reference on {_device_name()} ...", flush=True)
        t0 = time.time()
        res = score_rule(compile_rule(ADAM), task_names, use_fisher=False, seed=seed, steps_scale=steps_scale,
                         metric=metric, probe_lrs=probe_lrs)
        cache[key] = {n: dict(score=r["score"], lr=r["lr"]) for n, r in res.items()}
        clear_cache()
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        if verbose:
            print(f"[reference] done in {time.time() - t0:.0f}s: "
                  + ", ".join(f"{n}={v['score']:.4f}@lr={v['lr']:.2g}" for n, v in cache[key].items()),
                  flush=True)
    return {n: v["score"] for n, v in cache[key].items()}


class NNOptimizerSearch(BaseProblem):
    """Search element-wise optimizers for fast convergence.

    Args:
        use_fisher:   give the rule the Fisher sample f (variant R3).
        template:     'adam' or 'sgd' default body shown to the LLM.
        tasks:        task names from tasks.TASKS.
        steps_scale:  multiply every task's step count (smoke tests).
        log_dir:      directory receiving one JSON record per evaluation, or None.
        compute_reference: train the Adam reference on a cache miss (False in
                      the EoH parent, see reference_scores).
        aggregate:    'mean' or 'robust' (see aggregate_fitness).
        metric:       'train' (minibatch training loss) or 'val' (held-out loss; *_v tasks).
    With template='tensor' the rule is per-tensor and receives the run length T.
    """

    def __init__(self, use_fisher=False, template="adam", tasks=SEARCH_SUITE, seed=0,
                 steps_scale=1.0, log_dir=None, timeout=900, n_processes=1, compute_reference=True,
                 aggregate="mean", metric="train", probe_lrs=PROBE_LRS):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.use_fisher = use_fisher
        self.template_program = make_template(use_fisher, template)
        self.aggregate = aggregate
        self.metric = metric
        self.probe_lrs = tuple(probe_lrs)
        self.task_description = (make_tensor_description(aggregate) if template == "tensor"
                                 else make_description(use_fisher, aggregate))
        self.task_names = list(tasks)
        self.seed = seed
        self.steps_scale = steps_scale
        self.log_dir = log_dir
        self.reference = reference_scores(self.task_names, seed, steps_scale, compute=compute_reference,
                                          metric=metric, probe_lrs=self.probe_lrs)

    def evaluate_program(self, program_str, callable_func):
        t0 = time.time()
        record = dict(code_id=code_id(program_str), time=t0, pid=os.getpid(), code=program_str)
        speed = check_tensor_speed if takes_T(callable_func) else (lambda c: check_speed(c, self.use_fisher))
        reason = check_rule(callable_func, self.use_fisher) or speed(callable_func)
        if reason is not None:
            self._log(dict(record, fitness=None, rejected=reason, elapsed=time.time() - t0))
            return None
        try:
            # getattr: problem objects pickled by an older version of this file lack `metric`
            results = score_rule(callable_func, self.task_names, self.use_fisher, self.seed, self.steps_scale,
                                 metric=getattr(self, "metric", "train"),
                                 probe_lrs=getattr(self, "probe_lrs", PROBE_LRS))
        except Exception as e:  # LLM code may fail mid-training (dtype, device, shape, ...)
            self._log(dict(record, fitness=None, error=repr(e)[:2000], elapsed=time.time() - t0))
            return None
        rel = {n: r["score"] - self.reference[n] for n, r in results.items()}
        fitness = aggregate_fitness(rel, self.aggregate)
        self._log(dict(record, fitness=fitness, relative=rel, tasks=results, elapsed=time.time() - t0))
        return fitness

    def _log(self, record):
        """One file per evaluation (written atomically): evaluations run in
        parallel processes and the run directory lives on NFS, where
        concurrent appends to one file are not safe."""
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        name = f"{record['code_id']}_{record['pid']}_{int(record['time'] * 1000)}.json"
        tmp = os.path.join(self.log_dir, f".{name}.tmp")
        with open(tmp, "w") as f:
            json.dump(record, f)
        os.replace(tmp, os.path.join(self.log_dir, name))


def _print_baselines(use_fisher, task_names, steps_scale, out_path, metric="train", baselines=None,
                     probe_lrs=PROBE_LRS):
    reference = reference_scores(task_names, steps_scale=steps_scale, metric=metric, probe_lrs=probe_lrs)
    rows = {}
    for name, b in (baselines or baselines_for(use_fisher)).items():
        rule = compile_rule(b["code"])
        reason = check_rule(rule, use_fisher)
        if reason:
            print(f"{name:<18} REJECTED: {reason}")
            continue
        t0 = time.time()
        res = score_rule(rule, task_names, use_fisher, steps_scale=steps_scale, metric=metric, probe_lrs=probe_lrs)
        rel = {n: r["score"] - reference[n] for n, r in res.items()}
        rows[name] = dict(fitness=float(np.mean(list(rel.values()))), robust=aggregate_fitness(rel, "robust"),
                          relative=rel, lr={n: r["lr"] for n, r in res.items()}, tasks=res)
        print(f"{name:<18} mean={rows[name]['fitness']:+.4f} robust={rows[name]['robust']:+.4f}  "
              + "  ".join(f"{n}={rel[n]:+.3f}@{res[n]['lr']:.2g}" for n in task_names)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="compute the cached Adam reference")
    ap.add_argument("--baselines", action="store_true", help="score all hand-designed baselines")
    ap.add_argument("--fisher", action="store_true", help="Fisher-aware signature (variant 'fisher')")
    ap.add_argument("--r3", action="store_true", help="R3 setting: R3 tasks, validation loss, per-tensor baselines")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--steps-scale", type=float, default=1.0)
    args = ap.parse_args()
    tasks = args.tasks or (R3_SUITE if args.r3 else SEARCH_SUITE)
    metric = "val" if args.r3 else "train"

    probe_lrs = PROBE_LRS_R3 if args.r3 else PROBE_LRS
    if args.calibrate:
        print(reference_scores(tasks, steps_scale=args.steps_scale, metric=metric, probe_lrs=probe_lrs))
    if args.baselines:
        from baselines import TENSOR_BASELINES
        suffix = "r3" if args.r3 else ("fisher" if args.fisher else "plain")
        _print_baselines(args.fisher, tasks, args.steps_scale, os.path.join(HERE, f"baselines_{suffix}.json"),
                         metric=metric, baselines=TENSOR_BASELINES if args.r3 else None, probe_lrs=probe_lrs)
