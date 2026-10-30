"""Sanity checks for the harness and the theory, run before any EoH search.

    python selftest.py [--quick]

1. Baselines written as UpdateRules reproduce torch.optim.SGD / Adam exactly.
2. check_elementwise accepts every baseline and rejects rules that cheat
   (reductions, size/shape/position dependence, randomness).
3. P1: rule-level equivariance exponents are SGD -1, Adam / sign 0, diag NGD +1.
4. P2: E[f*f] matches the exact diagonal Fisher; B*g*g of the true-label batch
   gradient does not match the empirical Fisher.
5. The fitness of Adam itself is ~0 (it is the reference).
6. Per-tensor rules (run R3): Adam written per tensor reproduces the flat Adam
   exactly, the per-tensor checker accepts the per-tensor baselines and rejects
   cheats, and every R3 task reports a finite validation-loss curve.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "analysis"))

from baselines import ADAM, ADAM_T, SGD_MOMENTUM, TENSOR_BASELINES, baselines_for  # noqa: E402
from fisher import diag_fisher  # noqa: E402
from harness import (FlatModel, check_elementwise, check_tensor_rule, check_tensor_speed, compile_rule,  # noqa: E402
                     default_device, fisher_labels, run_rule)
from invariance import rule_exponent  # noqa: E402
from tasks import R3_SUITE, make_task  # noqa: E402

CHEATS = {
    "global norm": "return -lr * g / (g.norm() + 1e-8)",
    "tensor size": "return -lr * g / math.sqrt(p.numel())",
    "position": "return -lr * g * (1 + (torch.arange(p.numel(), device=p.device) % 2).reshape(p.shape))",
    "random": "return -lr * g * torch.rand_like(g)",
    "shape": "return -lr * g * (2 if p.dim() == 1 else 1)",
    "max": "return -lr * g / g.abs().max()",
    "scalar": "return torch.tensor(-lr)",
    "late reduction": "return -lr * g if t <= 10 else -lr * g / (g.norm() + 1e-8)",
}


def _report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def check_torch_optim(task_name, steps):
    device = default_device()
    ok = True
    for name, code, make_opt, lr in (
            ("SGD", SGD_MOMENTUM, lambda ps, lr: torch.optim.SGD(ps, lr, momentum=0.9), 1e-2),
            ("Adam", ADAM, lambda ps, lr: torch.optim.Adam(ps, lr), 1e-3)):
        task = make_task(task_name)
        ours = run_rule(task, compile_rule(code), lr, steps, seed=0, device=device)

        torch.manual_seed(0)
        model = task.build().to(device)
        opt = make_opt(model.parameters(), lr)
        gen = torch.Generator(device=device).manual_seed(0)
        ref = []
        for _ in range(steps):
            x, y = task.batch(gen)
            logits, targets = task.logits(model, x, y)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ref.append(loss.item())
        err = max(abs(a - b) / max(abs(b), 1e-8) for a, b in zip(ours, ref))
        ok &= _report(f"{name} rule matches torch.optim on {task_name}", err < 1e-4, f"(max rel err {err:.1e})")
    return ok


def check_checker():
    ok = True
    for fisher in (False, True):
        for name, b in baselines_for(fisher).items():
            reason = check_elementwise(compile_rule(b["code"]), fisher)
            ok &= _report(f"checker accepts {name}{' (f)' if fisher else ''}", reason is None, reason or "")
    for name, line in CHEATS.items():
        code = ("class UpdateRule:\n    def init_state(self, p):\n        return {}\n"
                f"    def step(self, p, g, state, t, lr):\n        {line}\n")
        reason = check_elementwise(compile_rule(code), False)
        ok &= _report(f"checker rejects {name}", reason is not None, f"({reason})")
    return ok


def check_p1():
    expected = dict(sgd_momentum=-1, adam=0, lion=0, signum=0, diag_ngd_emp=1)
    ok = True
    for name, k_exp in expected.items():
        k = rule_exponent(compile_rule(baselines_for(False)[name]["code"]), False, D=0.1)["median"]
        ok &= _report(f"P1 k({name}) = {k_exp:+d}", abs(k - k_exp) < 0.05, f"(measured {k:+.3f})")
    return ok


def check_p2(task_name, n_draws):
    device = default_device()
    task = make_task(task_name)
    task.setup(device)
    torch.manual_seed(0)
    flat = FlatModel(task.build().to(device))
    gen = torch.Generator(device=device).manual_seed(0)
    x, y = task.batch(gen)
    true, emp = diag_fisher(flat, x, y)

    theta = flat.theta
    est = torch.zeros_like(true)
    for _ in range(n_draws):
        logits = flat(x)
        g, = torch.autograd.grad(F.cross_entropy(logits, fisher_labels(logits, gen)), theta)
        est += x.shape[0] * g * g
    est /= n_draws
    g_true, = torch.autograd.grad(F.cross_entropy(flat(x), y), theta)
    bgg = x.shape[0] * g_true * g_true

    big = true > true.quantile(0.5)  # well-conditioned coordinates for a relative comparison
    rel_f = ((est - true).abs()[big] / true[big]).median().item()
    rel_g = ((bgg - emp).abs()[big] / emp[big].clamp_min(1e-30)).median().item()
    ok = _report("P2 E[f*f] = diag F (true Fisher)", rel_f < 0.15,
                 f"(median rel err {rel_f:.3f} with {n_draws} draws)")
    _report("P2 B*g*g != diag F_emp (true labels)", rel_g > 0.5, f"(median rel err {rel_g:.3f})")
    return ok


TENSOR_CHEATS = {
    "random": "return -lr * g * torch.rand_like(g)",
    "wrong shape": "return -lr * g.reshape(-1)",
    "nan": "return -lr * g / 0.0 * 0.0",
}


def check_tensor_rules(steps):
    ok = True
    a = run_rule(make_task("mlp_fmnist"), compile_rule(ADAM), 1e-3, steps)
    b = run_rule(make_task("mlp_fmnist"), compile_rule(ADAM_T), 1e-3, steps)
    err = float(abs(a - b).max() / abs(a).max())
    ok &= _report("per-tensor Adam = flat Adam on mlp_fmnist", err < 1e-5, f"(max rel err {err:.1e})")
    for name, b in TENSOR_BASELINES.items():
        cls = compile_rule(b["code"])
        reason = check_tensor_rule(cls) or check_tensor_speed(cls)
        ok &= _report(f"tensor checker accepts {name}", reason is None, reason or "")
    for name, line in TENSOR_CHEATS.items():
        code = ("import torch\nclass UpdateRule:\n    def init_state(self, p):\n        return {}\n"
                f"    def step(self, p, g, state, t, T, lr):\n        {line}\n")
        reason = check_tensor_rule(compile_rule(code))
        ok &= _report(f"tensor checker rejects {name}", reason is not None, f"({reason})")
    for task_name in R3_SUITE:
        task = make_task(task_name)
        v = run_rule(task, compile_rule(ADAM), 1e-3, 40, metric="val")
        ok &= _report(f"validation curve on {task_name}", bool((v == v).all()) and len(v) == 5,
                      f"(len {len(v)}, {v[0]:.3f} -> {v[-1]:.3f})")
    return ok


def check_reference(steps_scale):
    from prob import NNOptimizerSearch
    prob = NNOptimizerSearch(tasks=["softmax_illcond", "mlp_fmnist"], steps_scale=steps_scale)
    fit = prob.evaluate(ADAM)
    return _report("fitness(Adam) = 0", fit is not None and abs(fit) < 1e-6, f"({fit})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ok = True
    ok &= check_torch_optim("mlp_fmnist", 30 if args.quick else 100)
    ok &= check_checker()
    ok &= check_p1()
    ok &= check_p2("tiny_fmnist", 200 if args.quick else 1000)
    ok &= check_reference(0.25 if args.quick else 1.0)
    ok &= check_tensor_rules(30 if args.quick else 100)
    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
