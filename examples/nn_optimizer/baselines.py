"""Hand-designed optimizers written as UpdateRule sources.

The same sources serve as EoH seeds, as reference baselines in every results
table, and as fixtures for checking the harness against torch.optim.

Every rule is written for the plain signature step(self, p, g, state, t, lr);
fisher_signature() converts it to the Fisher-aware signature, which adds f
(a sample whose square is an unbiased estimate of the diagonal Fisher).
"""

SIG = "def step(self, p, g, state, t, lr):"
SIG_FISHER = "def step(self, p, g, f, state, t, lr):"

SGD_MOMENTUM = '''import torch
import math


class UpdateRule:
    def __init__(self):
        self.momentum = 0.9

    def init_state(self, p):
        return {"m": torch.zeros_like(p)}

    def step(self, p, g, state, t, lr):
        m = state["m"]
        m.mul_(self.momentum).add_(g)
        return -lr * m
'''

ADAM = '''import torch
import math


class UpdateRule:
    def __init__(self):
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def init_state(self, p):
        return {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}

    def step(self, p, g, state, t, lr):
        m, v = state["m"], state["v"]
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        return -lr * m_hat / (v_hat.sqrt() + self.eps)
'''

LION = '''import torch
import math


class UpdateRule:
    def __init__(self):
        self.beta1, self.beta2 = 0.9, 0.99

    def init_state(self, p):
        return {"m": torch.zeros_like(p)}

    def step(self, p, g, state, t, lr):
        m = state["m"]
        c = self.beta1 * m + (1 - self.beta1) * g
        m.mul_(self.beta2).add_(g, alpha=1 - self.beta2)
        return -lr * torch.sign(c)
'''

SIGNUM = '''import torch
import math


class UpdateRule:
    def __init__(self):
        self.beta = 0.9

    def init_state(self, p):
        return {"m": torch.zeros_like(p)}

    def step(self, p, g, state, t, lr):
        m = state["m"]
        m.mul_(self.beta).add_(g, alpha=1 - self.beta)
        return -lr * torch.sign(m)
'''

# Square-root-free diagonal preconditioner on the squared minibatch gradient
# (empirical-Fisher diagonal NGD / variational-online-Newton style).
# Damping chosen by a sweep over {1e-12, ..., 1e-2} on softmax_illcond,
# mlp_fmnist and tiny_fmnist; sqrt-free rules are very sensitive to it.
DIAG_NGD_EMP = '''import torch
import math


class UpdateRule:
    def __init__(self):
        self.beta1, self.beta2, self.damping = 0.9, 0.999, 1e-4

    def init_state(self, p):
        return {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}

    def step(self, p, g, state, t, lr):
        m, v = state["m"], state["v"]
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        return -lr * m_hat / (v_hat + self.damping)
'''

# Same, but the curvature is the true diagonal Fisher estimated from f * f.
# A single f * f is F * chi2_1 and often ~0, so early steps blow up unless the
# damping is large (same sweep as above; also beta2 in {0.99, 0.999}).
DIAG_NGD_FISHER = '''import torch
import math


class UpdateRule:
    def __init__(self):
        self.beta1, self.beta2, self.damping = 0.9, 0.99, 1e-2

    def init_state(self, p):
        return {"m": torch.zeros_like(p), "F": torch.zeros_like(p)}

    def step(self, p, g, f, state, t, lr):
        m, F = state["m"], state["F"]
        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
        F.mul_(self.beta2).addcmul_(f, f, value=1 - self.beta2)
        m_hat = m / (1 - self.beta1 ** t)
        F_hat = F / (1 - self.beta2 ** t)
        return -lr * m_hat / (F_hat + self.damping)
'''

BASELINES = {
    "sgd_momentum": dict(
        code=SGD_MOMENTUM,
        algorithm="Heavy-ball SGD: accumulate a momentum buffer m = 0.9 m + g and step along -lr * m."),
    "adam": dict(
        code=ADAM,
        algorithm="Adam: bias-corrected EMAs of the gradient and squared gradient; "
                  "step -lr * m_hat / (sqrt(v_hat) + eps)."),
    "lion": dict(
        code=LION,
        algorithm="Lion: step along the sign of an interpolation between momentum and the current "
                  "gradient, then update the momentum with a slower EMA."),
    "signum": dict(
        code=SIGNUM,
        algorithm="Signum: EMA momentum of the gradient, step -lr * sign(m)."),
    "diag_ngd_emp": dict(
        code=DIAG_NGD_EMP,
        algorithm="Square-root-free diagonal preconditioning: bias-corrected EMAs of g and g*g, "
                  "step -lr * m_hat / (v_hat + damping)."),
}

FISHER_BASELINES = {
    "diag_ngd_fisher": dict(
        code=DIAG_NGD_FISHER,
        algorithm="Square-root-free diagonal preconditioning with the Fisher sample: bias-corrected "
                  "EMAs of g and f*f, step -lr * m_hat / (F_hat + damping)."),
}


def fisher_signature(code):
    """Rewrite a plain-signature rule so it accepts (and ignores) f."""
    assert SIG in code, "rule does not use the plain step signature"
    return code.replace(SIG, SIG_FISHER)


def baselines_for(use_fisher):
    """All baselines as {name: dict(code, algorithm)} for the given signature."""
    if not use_fisher:
        return dict(BASELINES)
    out = {name: dict(b, code=fisher_signature(b["code"])) for name, b in BASELINES.items()}
    out.update(FISHER_BASELINES)
    return out


# ── per-tensor baselines with the run length T (run R3) ──────────────────────
# step(p, g, state, t, T, lr) is called for every parameter tensor separately, in
# its natural shape, with its own instance and state. The schedule helper is the
# "best practice" linear warmup (5% of T) followed by cosine decay to zero.

_SCHEDULE = '''
def schedule(t, T, warmup=0.05):
    w = max(1, int(warmup * T))
    if t <= w:
        return t / w
    return 0.5 * (1.0 + math.cos(math.pi * (t - w) / max(1, T - w)))
'''

ADAM_T = '''import torch
import math


class UpdateRule:
    """Adam with a constant learning rate, written for the per-tensor signature."""

    def init_state(self, p):
        return {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}

    def step(self, p, g, state, t, T, lr):
        m, v = state["m"], state["v"]
        m.mul_(0.9).add_(g, alpha=0.1)
        v.mul_(0.999).addcmul_(g, g, value=0.001)
        return -lr * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
'''

ADAMW_COSINE = '''import torch
import math
''' + _SCHEDULE + '''

class UpdateRule:
    """AdamW (decoupled weight decay 0.01) with linear warmup and cosine decay."""

    def init_state(self, p):
        return {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}

    def step(self, p, g, state, t, T, lr):
        m, v = state["m"], state["v"]
        m.mul_(0.9).add_(g, alpha=0.1)
        v.mul_(0.999).addcmul_(g, g, value=0.001)
        adam = (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
        return -lr * schedule(t, T) * (adam + 0.01 * p)
'''

SGD_COSINE = '''import torch
import math
''' + _SCHEDULE + '''

class UpdateRule:
    """Heavy-ball SGD (momentum 0.9) with linear warmup and cosine decay."""

    def init_state(self, p):
        return {"m": torch.zeros_like(p)}

    def step(self, p, g, state, t, T, lr):
        m = state["m"]
        m.mul_(0.9).add_(g)
        return -lr * schedule(t, T) * m
'''

LION_COSINE = '''import torch
import math
''' + _SCHEDULE + '''

class UpdateRule:
    """Lion (decoupled weight decay 0.01) with linear warmup and cosine decay."""

    def init_state(self, p):
        return {"m": torch.zeros_like(p)}

    def step(self, p, g, state, t, T, lr):
        m = state["m"]
        c = 0.9 * m + 0.1 * g
        m.mul_(0.99).add_(g, alpha=0.01)
        return -lr * schedule(t, T) * (torch.sign(c) + 0.01 * p)
'''

MUON_COSINE = '''import torch
import math
''' + _SCHEDULE + '''

def orthogonalize(G, steps=5):
    """Newton-Schulz iteration towards the polar factor U V^T of G (Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class UpdateRule:
    """Muon: Nesterov momentum, orthogonalized for matrices (conv kernels reshaped to
    out x fan-in) with RMS matched to Adam (0.2 * sqrt(max dim)); Adam for vectors.
    Linear warmup and cosine decay."""

    def init_state(self, p):
        return {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}

    def step(self, p, g, state, t, T, lr):
        m, v = state["m"], state["v"]
        if p.dim() >= 2:
            m.mul_(0.95).add_(g)
            G = (g + 0.95 * m).reshape(p.shape[0], -1)
            d = orthogonalize(G) * 0.2 * math.sqrt(max(G.shape))
            return -lr * schedule(t, T) * d.reshape(p.shape)
        m.mul_(0.9).add_(g, alpha=0.1)
        v.mul_(0.999).addcmul_(g, g, value=0.001)
        adam = (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
        return -lr * schedule(t, T) * adam
'''

SHAMPOO_COSINE = '''import torch
import math
''' + _SCHEDULE + '''

def inverse_root(A, k, rel_eps=1e-6):
    """A^(-1/k) of a symmetric PSD matrix in float64, damped by rel_eps * mean eigenvalue."""
    A = A.double()
    n = A.shape[0]
    A = A + rel_eps * (A.trace() / n).clamp_min(1e-30) * torch.eye(n, device=A.device, dtype=A.dtype)
    lam, U = torch.linalg.eigh(A)
    lam = lam.clamp_min(rel_eps * lam.max().clamp_min(1e-30))
    return ((U * lam.pow(-1.0 / k)) @ U.T).float()


def refresh(state, key):
    """Recompute the inverse root of state[key]; keep the old one if eigh fails."""
    try:
        state["P" + key] = inverse_root(state[key], 4)
    except RuntimeError:
        pass


class UpdateRule:
    """Shampoo: Kronecker-factored preconditioner L^-1/4 M R^-1/4 from EMAs of G G^T and
    G^T G (a side larger than 1024 is left unpreconditioned), inverse roots refreshed
    every 10 steps, step norm grafted from Adam per tensor; Adam for vectors. Linear
    warmup and cosine decay."""

    def init_state(self, p):
        st = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
        if p.dim() >= 2:
            rows, cols = p.shape[0], p[0].numel()
            for key, n in (("L", rows), ("R", cols)):
                if n <= 1024:
                    st[key] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
                    st["P" + key] = torch.eye(n, device=p.device, dtype=p.dtype)
        return st

    def step(self, p, g, state, t, T, lr):
        m, v = state["m"], state["v"]
        m.mul_(0.9).add_(g, alpha=0.1)
        v.mul_(0.999).addcmul_(g, g, value=0.001)
        adam = (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
        if p.dim() < 2:
            return -lr * schedule(t, T) * adam
        G = g.reshape(p.shape[0], -1)
        D = (m / (1 - 0.9 ** t)).reshape(G.shape)
        if "L" in state:
            state["L"].mul_(0.95).add_(G @ G.T, alpha=0.05)
            if t % 10 == 1:
                refresh(state, "L")
            D = state["PL"] @ D
        if "R" in state:
            state["R"].mul_(0.95).add_(G.T @ G, alpha=0.05)
            if t % 10 == 1:
                refresh(state, "R")
            D = D @ state["PR"]
        D = D * (adam.norm() / D.norm().clamp_min(1e-30))
        return -lr * schedule(t, T) * D.reshape(p.shape)
'''

TENSOR_BASELINES = {
    "adam_t": dict(code=ADAM_T, algorithm="Adam with a constant learning rate."),
    "adamw_cosine": dict(code=ADAMW_COSINE, algorithm="AdamW with linear warmup and cosine decay."),
    "sgd_cosine": dict(code=SGD_COSINE, algorithm="Heavy-ball SGD with linear warmup and cosine decay."),
    "lion_cosine": dict(code=LION_COSINE, algorithm="Lion with weight decay, linear warmup and cosine decay."),
    "muon_cosine": dict(code=MUON_COSINE, algorithm="Muon (orthogonalized momentum for matrices, Adam for "
                                                     "vectors) with linear warmup and cosine decay."),
    "shampoo_cosine": dict(code=SHAMPOO_COSINE, algorithm="Shampoo (Kronecker-factored L^-1/4 M R^-1/4, Adam "
                                                          "grafting) with linear warmup and cosine decay."),
}
