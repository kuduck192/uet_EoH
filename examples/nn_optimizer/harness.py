"""Training harness for evolved element-wise update rules.

An UpdateRule is applied to the *flattened* parameter vector of a model in a
single call per step. This is exactly equivalent to applying it per tensor
because check_elementwise() rejects any rule whose output at one coordinate
depends on other coordinates, on the tensor shape or on the tensor size.

Losses are minibatch training losses recorded before each update; every
candidate sees the same initialisation and batch sequence for a given seed.
"""

import ast
import inspect
import math
import os
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

RULE_PREAMBLE = "import torch\nimport math\n"

# Log-spaced learning-rate probe followed by a local refinement; wide enough
# that rules with very different natural step scales (SGD ~1e-1, Adam ~1e-3,
# sqrt-free preconditioners ~1e-5) are all tuned fairly.
PROBE_LRS = tuple(10.0 ** k for k in range(-5, 2))
PROBE_FRAC = 0.25
REFINE = (1 / 3, 1.0, 3.0)
DIVERGENCE_FACTOR = 10.0


def default_device():
    """$NNOPT_DEVICE, or cuda / cpu. NNOPT_DEVICE=rr spreads processes over all
    visible GPUs by pid (each EoH evaluation runs in its own process)."""
    name = os.environ.get("NNOPT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if name == "rr":
        name = f"cuda:{os.getpid() % torch.cuda.device_count()}"
    return torch.device(name)


def compile_rule(code, entry="UpdateRule"):
    """Exec a rule's source and return its class."""
    module = types.ModuleType("rule_module")
    exec(RULE_PREAMBLE, module.__dict__)
    exec(code, module.__dict__)
    return getattr(module, entry)


def strip_docstrings(tree):
    """Remove module, class and function docstrings from an AST (in place)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


class _ZeroNumbers(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(0), node)
        return node


def code_skeleton(code):
    """Structure of a rule without docstrings, comments and numeric constants:
    rules that differ only in hyper-parameter values share a skeleton."""
    try:
        tree = strip_docstrings(ast.parse(code))
    except SyntaxError:
        return code
    return ast.dump(_ZeroNumbers().visit(tree))


class FlatModel:
    """Expose a module's parameters as one flat vector theta (the optimizer variable).

    With param_scale = D the optimizer works in phi = D * theta coordinates:
    the module sees theta = phi / D. Used by the reparametrization test.
    """

    def __init__(self, model, param_scale=None):
        self.model = model
        named = list(model.named_parameters())
        self.names = [n for n, _ in named]
        self.shapes = [p.shape for _, p in named]
        self.numels = [p.numel() for _, p in named]
        theta = torch.cat([p.detach().reshape(-1) for _, p in named])
        self.scale = param_scale
        if param_scale is not None:
            theta = theta * param_scale
        self.theta = theta.clone().requires_grad_(True)

    def slices(self):
        out, start = {}, 0
        for name, n in zip(self.names, self.numels):
            out[name] = slice(start, start + n)
            start += n
        return out

    def params(self, theta=None):
        theta = self.theta if theta is None else theta
        if self.scale is not None:
            theta = theta / self.scale
        return {n: t.view(s) for n, t, s in zip(self.names, theta.split(self.numels), self.shapes)}

    def __call__(self, x, theta=None):
        return functional_call(self.model, self.params(theta), (x,))


def fisher_labels(logits, gen):
    """Sample y ~ softmax(logits) with the Gumbel-max trick (robust to non-finite logits)."""
    u = torch.rand(logits.shape, generator=gen, device=logits.device).clamp_(1e-20, 1.0)
    return (logits.detach() - torch.log(-torch.log(u))).argmax(-1)


def takes_T(rule_cls):
    """True for per-tensor rules, whose step(p, g, state, t, T, lr) also receives the
    run length T; they are applied to every parameter tensor separately."""
    return "T" in inspect.signature(rule_cls.step).parameters


def run_rule(task, rule_cls, lr, steps=None, seed=0, use_fisher=False, device=None,
             hook=None, param_scale=None, metric="train", val_every=10):
    """Train task with rule_cls at learning rate lr.

    Returns a float64 array of losses; entries after a divergence are NaN. With
    metric="train" these are the per-step minibatch losses; with metric="val"
    the held-out losses at t = 1 and every val_every steps (before the update).
    Element-wise rules see the flattened parameter vector; per-tensor rules
    (takes_T) get one instance and one state per parameter tensor.
    hook(t, info) is called every step before the update is applied.
    param_scale(n_params, device) -> D optionally makes the optimizer run in
    phi = D * theta coordinates (reparametrization test).
    """
    device = device or default_device()
    steps = steps or task.steps
    if getattr(task, "device", None) != device:
        task.setup(device)
    torch.manual_seed(seed)
    model = task.build().to(device)
    flat = FlatModel(model, None if param_scale is None else param_scale(flat_size(model), device))
    theta = flat.theta
    gen = torch.Generator(device=device).manual_seed(seed)
    fgen = torch.Generator(device=device).manual_seed(seed + 10_000)

    per_tensor = takes_T(rule_cls)
    if per_tensor:
        parts = list(zip(flat.slices().values(), flat.shapes))
        rules = [rule_cls() for _ in parts]
        states = [r.init_state(theta.detach()[sl].view(sh).clone()) for r, (sl, sh) in zip(rules, parts)]
        state = states
    else:
        rule = rule_cls()
        state = rule.init_state(theta.detach().clone())
    losses = torch.full((steps,), float("nan"), device=device, dtype=torch.float64)
    val = torch.full((steps // val_every + 1,), float("nan"), device=device, dtype=torch.float64) \
        if metric == "val" else None
    l0 = None

    for t in range(1, steps + 1):
        x, y = task.batch(gen)
        logits, targets = task.logits(flat, x, y)
        loss = F.cross_entropy(logits, targets)
        if use_fisher:
            loss_f = F.cross_entropy(logits, fisher_labels(logits, fgen))
            g, = torch.autograd.grad(loss, theta, retain_graph=True)
            gf, = torch.autograd.grad(loss_f, theta)
            # Per-sample scores have zero mean under the model, so cross terms
            # vanish: E[f * f] = mean per-sample diagonal Fisher.
            f = gf * math.sqrt(targets.shape[0])
            delta = rule.step(theta.detach().clone(), g.clone(), f.clone(), state, t, lr)
        elif per_tensor:
            g, = torch.autograd.grad(loss, theta)
            f = None
            th, out = theta.detach(), []
            for r, st, (sl, sh) in zip(rules, states, parts):
                d = r.step(th[sl].view(sh).clone(), g[sl].view(sh).clone(), st, t, steps, lr)
                if not torch.is_tensor(d) or d.shape != sh:
                    raise ValueError(f"step() must return a tensor of shape {tuple(sh)}")
                out.append(d.reshape(-1))
            delta = torch.cat(out)
        else:
            g, = torch.autograd.grad(loss, theta)
            f = None
            delta = rule.step(theta.detach().clone(), g.clone(), state, t, lr)

        if not torch.is_tensor(delta) or delta.shape != theta.shape:
            raise ValueError(f"step() must return a tensor of shape {tuple(theta.shape)}")
        if val is not None and (t == 1 or t % val_every == 0):
            val[t // val_every] = task.val_loss(flat)
        if hook is not None:
            hook(t, dict(flat=flat, theta=theta.detach(), g=g, f=f, delta=delta, state=state,
                         loss=loss.detach(), x=x, y=y, task=task))
        with torch.no_grad():
            theta.add_(delta.to(device=theta.device, dtype=theta.dtype))
        losses[t - 1] = loss.detach()

        if t == 1:
            l0 = float(losses[0])
        if t % 25 == 0 or t == steps:
            recent = losses[max(0, t - 25):t]
            if not bool(torch.isfinite(recent).all()) or float(recent.max()) > DIVERGENCE_FACTOR * l0:
                break

    return (val if val is not None else losses).cpu().numpy()


def flat_size(model):
    return sum(p.numel() for p in model.parameters())


def curve_score(losses):
    """Mean log loss over the run (log-AUC). Divergence counts as no progress.

    Losses are clipped at the initial loss, and steps after a divergence are
    filled with it, so the score is bounded by log(L0).
    """
    losses = np.asarray(losses, dtype=float)
    l0 = losses[0]
    if not np.isfinite(l0):
        return float("inf")
    bad = ~np.isfinite(losses)
    if bad.any():
        losses = losses.copy()
        losses[np.argmax(bad):] = l0
    return float(np.mean(np.log(np.clip(losses, 1e-12, l0))))


def tune_lr(task, rule_cls, seed=0, use_fisher=False, device=None,
            probe_lrs=PROBE_LRS, probe_frac=PROBE_FRAC, refine=REFINE, param_scale=None, metric="train"):
    """Probe a wide log grid on a short horizon, then refine around the best lr.

    Returns dict(score, lr, losses, probe, full) where score is the best
    full-horizon curve_score. param_scale is passed on to run_rule.
    """
    device = device or default_device()
    probe_steps = max(20, int(task.steps * probe_frac))
    probe = {}
    for lr in probe_lrs:
        probe[lr] = curve_score(run_rule(task, rule_cls, lr, probe_steps, seed, use_fisher, device,
                                         param_scale=param_scale, metric=metric))
    best_probe = min(probe, key=probe.get)

    full, curves = {}, {}
    for factor in refine:
        lr = best_probe * factor
        curves[lr] = run_rule(task, rule_cls, lr, task.steps, seed, use_fisher, device, param_scale=param_scale,
                              metric=metric)
        full[lr] = curve_score(curves[lr])
    best = min(full, key=full.get)
    return dict(score=full[best], lr=best, losses=curves[best].tolist(),
                probe={f"{k:.3g}": v for k, v in probe.items()},
                full={f"{k:.3g}": v for k, v in full.items()})


# ── element-wise check ────────────────────────────────────────────────────────

def check_elementwise(rule_cls, use_fisher=False, device=None, n=128, n_steps=4, lr=1e-2):
    """Return None if rule_cls is a deterministic coordinate-wise rule, else a reason.

    The update of coordinate i may depend only on (p_i, g_i[, f_i], state_i, t, lr).
    Tested by comparing runs on the same coordinate inputs embedded in tensors
    of other shapes, sizes and orderings, both for the first steps and for a
    sequence starting at t = 1000 (rules that branch on t cannot hide there).
    """
    device = device or default_device()
    gen = torch.Generator().manual_seed(1234)

    def rand(size, scale=1.0):
        mag = 10.0 ** (torch.rand(size, generator=gen) * 5 - 4)  # magnitudes in [1e-4, 10]
        return (scale * mag * torch.randn(size, generator=gen)).to(device)

    def make_seq(size, scale=1.0):
        return [dict(p=rand(size, scale), g=rand(size, scale), f=rand(size, scale)) for _ in range(n_steps)]

    base_seq = make_seq(n)
    extra_seq = make_seq(n, scale=100.0)

    def run(seq, transform, inverse, t0):
        rule = rule_cls()
        state = rule.init_state(transform(seq[0]["p"]).clone())
        outs = []
        for t, inp in enumerate(seq, start=t0):
            args = [transform(inp["p"]).clone(), transform(inp["g"]).clone()]
            if use_fisher:
                args.append(transform(inp["f"]).clone())
            delta = rule.step(*args, state, t, lr)
            if not torch.is_tensor(delta):
                raise TypeError("step() must return a torch.Tensor")
            if delta.shape != args[0].shape:
                raise ValueError(f"delta shape {tuple(delta.shape)} != parameter shape {tuple(args[0].shape)}")
            outs.append(inverse(delta.detach()).reshape(-1).double().cpu())
        return torch.stack(outs)

    def same(a, b):
        return torch.allclose(a, b, rtol=1e-5, atol=1e-10, equal_nan=True)

    ident = lambda v: v
    ext_seq = [{k: torch.cat([b[k], e[k]]) for k in b} for b, e in zip(base_seq, extra_seq)]
    perm = torch.randperm(n, generator=gen).to(device)
    inv = torch.argsort(perm)
    try:
        for t0 in (1, 1000):
            at = "" if t0 == 1 else f" (steps t >= {t0})"
            base = run(base_seq, ident, ident, t0)
            if not torch.isfinite(base).all():
                return "update is not finite on typical inputs" + at
            if not same(run(base_seq, ident, ident, t0), base):
                return "update is not deterministic" + at
            if not same(run(base_seq, lambda v: v.reshape(8, n // 8), ident, t0), base):
                return "update depends on the tensor shape" + at
            if not same(run(base_seq, lambda v: v[perm], lambda v: v.reshape(-1)[inv], t0), base):
                return "update depends on coordinate positions" + at
            if not same(run(ext_seq, ident, lambda v: v[:n], t0), base):
                return "update of a coordinate depends on other coordinates (reduction over elements)" + at
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def check_speed(rule_cls, use_fisher=False, device=None, n=2 ** 20, max_seconds=0.2):
    """Return None if one step on n coordinates takes at most max_seconds, else a reason.

    Rejects per-element Python loops and similar code early, before they spend
    a whole evaluation timeout (the search models have ~1e5 parameters).
    """
    device = device or default_device()
    gen = torch.Generator().manual_seed(4321)
    p, g, f = (0.1 * torch.randn(n, generator=gen).to(device) for _ in range(3))
    try:
        rule = rule_cls()
        state = rule.init_state(p.clone())
        args = (p.clone(), g.clone(), f.clone()) if use_fisher else (p.clone(), g.clone())
        rule.step(*args, state, 1, 1e-3)  # warm-up (kernel compilation, allocations)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        for t in range(2, 5):
            rule.step(*args, state, t, 1e-3)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_step = (time.time() - t0) / 3
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if per_step > max_seconds:
        return f"too slow: {per_step:.2f}s per step on {n} parameters"
    return None


def check_rule(rule_cls, use_fisher=False):
    """check_tensor_rule for per-tensor rules (takes_T), check_elementwise otherwise."""
    if takes_T(rule_cls):
        return check_tensor_rule(rule_cls)
    return check_elementwise(rule_cls, use_fisher)


CHECK_SHAPES = [(48,), (24, 40), (40, 24), (8, 4, 3, 3), (10, 64)]


def check_tensor_rule(rule_cls, device=None, n_steps=4):
    """Return None if a per-tensor rule is deterministic, keeps shapes and gives finite
    updates on tensors of several ranks (early steps and t >= 1000), else a reason."""
    device = device or default_device()

    def run(shape, t0, T):
        gen = torch.Generator().manual_seed(77)
        p = (0.1 * torch.randn(shape, generator=gen)).to(device)
        gs = [(0.01 * torch.randn(shape, generator=gen)).to(device) for _ in range(n_steps)]
        rule = rule_cls()
        state = rule.init_state(p.clone())
        outs = []
        for i, g in enumerate(gs):
            d = rule.step(p.clone(), g.clone(), state, t0 + i, T, 1e-3)
            if not torch.is_tensor(d):
                raise TypeError("step() must return a torch.Tensor")
            if d.shape != p.shape:
                raise ValueError(f"delta shape {tuple(d.shape)} != parameter shape {tuple(p.shape)}")
            outs.append(d.detach().double().reshape(-1).cpu())
            p = p + d
        return torch.cat(outs)

    try:
        for t0, T in ((1, 100), (1000, 1100)):
            for shape in CHECK_SHAPES:
                a = run(shape, t0, T)
                if not torch.isfinite(a).all():
                    return f"update is not finite for a tensor of shape {shape}"
                if not torch.allclose(a, run(shape, t0, T), rtol=1e-5, atol=1e-10):
                    return "update is not deterministic"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


SPEED_SHAPES = [(128, 784), (128,), (128, 128), (10, 128), (64, 32, 3, 3), (192, 64), (65, 64)]


def check_tensor_speed(rule_cls, device=None, max_seconds=0.5):
    """Reject per-tensor rules whose step over a model-sized set of tensors takes
    longer than max_seconds (averaged over 5 steps after a warm-up step)."""
    device = device or default_device()
    gen = torch.Generator().manual_seed(4321)
    try:
        ps = [(0.1 * torch.randn(s, generator=gen)).to(device) for s in SPEED_SHAPES]
        rules = [rule_cls() for _ in ps]
        states = [r.init_state(p.clone()) for r, p in zip(rules, ps)]

        def one(t):
            for r, st, p in zip(rules, states, ps):
                r.step(p.clone(), 0.01 * p, st, t, 1000, 1e-3)

        one(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        for t in range(2, 7):
            one(t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_step = (time.time() - t0) / 5
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if per_step > max_seconds:
        return f"too slow: {per_step:.2f}s per step on a model-sized set of tensors"
    return None


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
