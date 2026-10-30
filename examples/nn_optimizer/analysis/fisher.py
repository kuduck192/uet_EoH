"""Fisher-information utilities for flat-parameter models.

For softmax cross-entropy the true Fisher equals the generalised Gauss-Newton
matrix, F = E_x[J^T (diag(p) - p p^T) J] with J = d logits / d theta, so the
"true Fisher" below is also the GGN. Per-example quantities (diag_fisher,
full_fisher) assume a classification task (one label per example);
mc_diag_fisher works for any task, including the sequence model.
"""

import torch
import torch.nn.functional as F
from torch.func import grad, jacrev, vmap

import common  # noqa: F401  (puts the example root on sys.path)
from harness import fisher_labels


def _logits_one(flat):
    return lambda theta, xi: flat(xi[None], theta)[0]


def per_sample_grads(flat, x, labels, chunk=128):
    """(N, P) per-example gradients of the CE loss for the given labels."""
    logits_one = _logits_one(flat)

    def loss_fn(theta, xi, yi):
        return F.cross_entropy(logits_one(theta, xi)[None], yi[None])

    theta = flat.theta.detach()
    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    return torch.cat([g(theta, x[i:i + chunk], labels[i:i + chunk]) for i in range(0, x.shape[0], chunk)])


def diag_fisher(flat, x, y, chunk=128):
    """Exact diagonals of the true Fisher (expectation over all classes under the
    model) and of the empirical Fisher (true labels) on the batch. Returns (true, emp)."""
    with torch.no_grad():
        probs = flat(x).softmax(-1)
    true = torch.zeros_like(flat.theta.detach())
    for c in range(probs.shape[1]):
        gc = per_sample_grads(flat, x, torch.full_like(y, c), chunk)
        true += (probs[:, c:c + 1] * gc * gc).sum(0)
    true /= x.shape[0]
    ge = per_sample_grads(flat, x, y, chunk)
    return true, (ge * ge).mean(0)


def mc_diag_fisher(flat, task, x, y, gen, n_draws=16):
    """Monte-Carlo true diagonal Fisher from batch Fisher samples, E[N * g_hat^2] = diag F."""
    theta = flat.theta.detach().clone().requires_grad_(True)
    out = torch.zeros_like(theta)
    for _ in range(n_draws):
        logits, targets = task.logits(lambda inp: flat(inp, theta), x, y)
        g, = torch.autograd.grad(F.cross_entropy(logits, fisher_labels(logits, gen)), theta)
        out += targets.shape[0] * g * g
    return out / n_draws


def full_fisher(flat, x, chunk=128):
    """Exact (P, P) true Fisher / GGN on the batch. Only for small models."""
    logits_one = _logits_one(flat)
    theta = flat.theta.detach()
    jac = vmap(jacrev(logits_one), in_dims=(None, 0))
    P = theta.numel()
    out = torch.zeros(P, P, device=theta.device, dtype=theta.dtype)
    for i in range(0, x.shape[0], chunk):
        xi = x[i:i + chunk]
        J = jac(theta, xi)                                       # (n, C, P)
        with torch.no_grad():
            p = flat(xi, theta).softmax(-1)                      # (n, C)
        H = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]  # (n, C, C)
        out += torch.einsum("ncp,ncd,ndq->pq", J, H, J)
    return out / x.shape[0]
