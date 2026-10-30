"""Training tasks for optimizer discovery.

Each task trains a small model from a fixed initialisation on a fixed batch
sequence (common random numbers), so all candidate optimizers are compared on
exactly the same problem instance. Every task exposes logits of shape (N, C)
and integer targets of shape (N,), so the harness can treat them uniformly
(cross-entropy loss, Fisher sampling from the predictive distribution).

Datasets are read from $NNOPT_DATA (default: ./data) as produced by
prepare_data.py and kept on the GPU; no DataLoader is involved.
"""

import os

import torch
import torch.nn as nn

DATA_DIR = os.environ.get(
    "NNOPT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)

_DATA_CACHE = {}


def load_dataset(name, device):
    key = (name, str(device))
    if key not in _DATA_CACHE:
        path = os.path.join(DATA_DIR, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run prepare_data.py first")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        _DATA_CACHE[key] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in blob.items()}
    return _DATA_CACHE[key]


def clear_cache():
    _DATA_CACHE.clear()


class Task:
    """A training problem: build() a model, sample batches, map model output to logits.

    Tasks constructed with val=True also hold a fixed held-out set (self.xv,
    self.yv) for val_loss(); their names end in "_v".
    """

    name = ""
    steps = 0
    batch_size = 0
    val = False

    def setup(self, device):
        self.device = device

    def val_loss(self, model):
        """Mean loss on the held-out set (no gradient)."""
        with torch.no_grad():
            logits, targets = self.logits(model, self.xv, self.yv)
            return torch.nn.functional.cross_entropy(logits, targets)

    def build(self) -> nn.Module:
        raise NotImplementedError

    def batch(self, gen):
        """Return (x, y) for one minibatch drawn with the device generator gen."""
        raise NotImplementedError

    def logits(self, model, x, y):
        """Return (logits (N, C), targets (N,))."""
        return model(x), y

    def _randint(self, high, gen):
        return torch.randint(high, (self.batch_size,), generator=gen, device=self.device)


# ── T1: axis-aligned ill-conditioned softmax regression ──────────────────────

class SoftmaxIllCond(Task):
    """Linear softmax classifier on features with log-spaced scales.

    Labels come from a teacher on the unscaled features, so the input
    covariance (and hence the Hessian/Fisher) is diagonal with condition number
    ~ 10**(2 * log10_scale). Diagonal preconditioning can fix this exactly.
    With rotate=True the features are multiplied by a random orthogonal matrix:
    same spectrum, but no longer axis-aligned, so element-wise rules lose
    their advantage while full natural gradient is unaffected.
    """

    name = "softmax_illcond"

    def __init__(self, steps=300, batch_size=256, dim=64, classes=10, n=8192, log10_scale=1.5,
                 rotate=False, val=False, n_val=4096):
        self.steps, self.batch_size = steps, batch_size
        self.dim, self.classes, self.n, self.log10_scale = dim, classes, n, log10_scale
        self.rotate, self.val, self.n_val = rotate, val, n_val
        if rotate:
            self.name = "softmax_illcond_rot"

    def setup(self, device):
        super().setup(device)
        g = torch.Generator().manual_seed(0)
        z = torch.randn(self.n, self.dim, generator=g)
        w = torch.randn(self.dim, self.classes, generator=g)
        y = (z @ w + 0.5 * torch.randn(self.n, self.classes, generator=g)).argmax(1)
        scales = torch.logspace(0, self.log10_scale, self.dim)
        x = z * scales
        if self.rotate:
            q, _ = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=g))
            x = x @ q
        self.x = x.to(device)
        self.y = y.to(device)
        if self.val:  # same distribution, drawn after the training set
            zv = torch.randn(self.n_val, self.dim, generator=g)
            yv = (zv @ w + 0.5 * torch.randn(self.n_val, self.classes, generator=g)).argmax(1)
            xv = zv * scales
            if self.rotate:
                xv = xv @ q
            self.xv, self.yv = xv.to(device), yv.to(device)

    def build(self):
        m = nn.Linear(self.dim, self.classes)
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        return m

    def batch(self, gen):
        idx = self._randint(self.n, gen)
        return self.x[idx], self.y[idx]


# ── T2: MLP on Fashion-MNIST ──────────────────────────────────────────────────

class MLPFashion(Task):
    name = "mlp_fmnist"
    MEAN, STD = 0.2860, 0.3530

    def __init__(self, steps=400, batch_size=128, width=128, val=False, n_val=4096):
        self.steps, self.batch_size, self.width = steps, batch_size, width
        self.val, self.n_val = val, n_val

    def _norm(self, x):
        return (x.float() / 255.0 - self.MEAN) / self.STD

    def setup(self, device):
        super().setup(device)
        d = load_dataset("fmnist", device)
        self.x, self.y = d["x"], d["y"]
        if self.val:
            t = load_dataset("fmnist_test", device)
            self.xv, self.yv = self._norm(t["x"][:self.n_val]), t["y"][:self.n_val]

    def build(self):
        w = self.width
        return nn.Sequential(
            nn.Linear(784, w), nn.ReLU(),
            nn.Linear(w, w), nn.ReLU(),
            nn.Linear(w, 10),
        )

    def batch(self, gen):
        idx = self._randint(self.x.shape[0], gen)
        return self._norm(self.x[idx]), self.y[idx]


class TinyFashion(MLPFashion):
    """Fashion-MNIST average-pooled to 7x7 with a 49-32-10 MLP (~2k parameters),
    small enough to form the exact Fisher matrix (analysis/ngd_alignment.py)."""

    name = "tiny_fmnist"

    def __init__(self, steps=400, batch_size=128, width=32):
        super().__init__(steps, batch_size, width)

    def build(self):
        return nn.Sequential(nn.Linear(49, self.width), nn.Tanh(), nn.Linear(self.width, 10))

    def batch(self, gen):
        x, y = super().batch(gen)
        x = torch.nn.functional.avg_pool2d(x.view(-1, 1, 28, 28), 4).reshape(-1, 49)
        return x, y


# ── T3: small CNN on CIFAR-10 ────────────────────────────────────────────────

class _CifarBatches(Task):
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    n_val = 2048

    def setup(self, device):
        super().setup(device)
        d = load_dataset("cifar10", device)
        self.x, self.y = d["x"], d["y"]
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)
        if self.val:
            t = load_dataset("cifar10_test", device)
            self.xv = (t["x"][:self.n_val].float() / 255.0 - self.mean) / self.std
            self.yv = t["y"][:self.n_val]

    def batch(self, gen):
        idx = self._randint(self.x.shape[0], gen)
        x = (self.x[idx].float() / 255.0 - self.mean) / self.std
        return x, self.y[idx]


class CNNCifar(_CifarBatches):
    name = "cnn_cifar"

    def __init__(self, steps=400, batch_size=128, val=False):
        self.steps, self.batch_size, self.val = steps, batch_size, val

    def build(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 10),
        )


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.short(x))


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [_BasicBlock(cin, cout, stride), _BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(512, classes)

    def forward(self, x):
        h = self.layers(self.stem(x))
        return self.head(h.mean(dim=(2, 3)))


class ResNetCifar(_CifarBatches):
    name = "resnet18_cifar"

    def __init__(self, steps=2000, batch_size=128, val=False):
        self.steps, self.batch_size, self.val = steps, batch_size, val

    def build(self):
        return ResNet18()


# ── T4: character-level transformer on TinyShakespeare ───────────────────────

class CharTransformer(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, ctx):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.02 * torch.randn(ctx, d))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(ctx), persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        h = self.tok(idx) + self.pos[:T]
        for block in self.blocks:
            h = block(h, src_mask=self.mask[:T, :T], is_causal=True)
        return self.head(self.ln(h))


class CharLM(Task):
    name = "charlm_shakespeare"

    def __init__(self, steps=400, batch_size=32, d=64, n_layer=2, n_head=4, ctx=64, name=None, val=False,
                 n_val_seq=64):
        self.steps, self.batch_size = steps, batch_size
        self.d, self.n_layer, self.n_head, self.ctx = d, n_layer, n_head, ctx
        self.val, self.n_val_seq = val, n_val_seq
        if name:
            self.name = name

    def setup(self, device):
        super().setup(device)
        d = load_dataset("shakespeare", device)
        self.tokens = d["tokens"].long()
        self.vocab = int(d["vocab_size"])
        self.offsets = torch.arange(self.ctx + 1, device=device)
        if self.val:  # train on the first 90% of the text, validate on fixed windows of the rest
            cut = int(0.9 * self.tokens.shape[0])
            held = self.tokens[cut:]
            self.tokens = self.tokens[:cut]
            starts = torch.linspace(0, held.shape[0] - self.ctx - 2, self.n_val_seq, device=device).long()
            chunk = held[starts[:, None] + self.offsets]
            self.xv, self.yv = chunk[:, :-1], chunk[:, 1:]

    def build(self):
        return CharTransformer(self.vocab, self.d, self.n_layer, self.n_head, self.ctx)

    def batch(self, gen):
        start = self._randint(self.tokens.shape[0] - self.ctx - 1, gen)
        chunk = self.tokens[start[:, None] + self.offsets]
        return chunk[:, :-1], chunk[:, 1:]

    def logits(self, model, x, y):
        out = model(x)
        return out.reshape(-1, out.shape[-1]), y.reshape(-1)


# ── registries ───────────────────────────────────────────────────────────────

TASKS = {
    # search suite
    "softmax_illcond": lambda: SoftmaxIllCond(),
    "mlp_fmnist": lambda: MLPFashion(),
    "cnn_cifar": lambda: CNNCifar(),
    "charlm_shakespeare": lambda: CharLM(),
    # held-out transfer suite: larger models, other batch sizes, 10x horizon
    "resnet18_cifar": lambda: ResNetCifar(),
    "gpt_shakespeare": lambda: CharLM(steps=2000, batch_size=64, d=256, n_layer=6, n_head=8, ctx=256,
                                      name="gpt_shakespeare"),
    "mlp_fmnist_long": lambda: _renamed(MLPFashion(steps=4000), "mlp_fmnist_long"),
    "cnn_cifar_bs512": lambda: _renamed(CNNCifar(steps=1000, batch_size=512), "cnn_cifar_bs512"),
    "softmax_illcond_long": lambda: _renamed(SoftmaxIllCond(steps=3000), "softmax_illcond_long"),
    # analysis-only tasks
    "softmax_illcond_rot": lambda: SoftmaxIllCond(rotate=True),
    "tiny_fmnist": lambda: TinyFashion(),
}

SEARCH_SUITE = ["softmax_illcond", "mlp_fmnist", "cnn_cifar", "charlm_shakespeare"]
TRANSFER_SUITE = ["resnet18_cifar", "gpt_shakespeare", "mlp_fmnist_long", "cnn_cifar_bs512",
                  "softmax_illcond_long"]

# Run R3: validation-loss tasks with different horizons T (a rule cannot hard-code a
# step count), and the *rotated* ill-conditioned problem, which diagonal rules
# cannot precondition (rotation test) but matrix preconditioners can.
TASKS.update({
    "softmax_rot_v": lambda: _renamed(SoftmaxIllCond(steps=150, rotate=True, val=True), "softmax_rot_v"),
    "mlp_fmnist_v": lambda: _renamed(MLPFashion(steps=400, val=True), "mlp_fmnist_v"),
    "cnn_cifar_v": lambda: _renamed(CNNCifar(steps=200, val=True), "cnn_cifar_v"),
    "charlm_v": lambda: CharLM(steps=300, val=True, name="charlm_v"),
    # held-out
    "softmax_rot_long_v": lambda: _renamed(SoftmaxIllCond(steps=2000, rotate=True, val=True), "softmax_rot_long_v"),
    "mlp_fmnist_long_v": lambda: _renamed(MLPFashion(steps=3000, val=True), "mlp_fmnist_long_v"),
    "cnn_cifar_bs512_v": lambda: _renamed(CNNCifar(steps=1000, batch_size=512, val=True), "cnn_cifar_bs512_v"),
    "resnet18_cifar_v": lambda: _renamed(ResNetCifar(val=True), "resnet18_cifar_v"),
    "gpt_v": lambda: CharLM(steps=2000, batch_size=64, d=256, n_layer=6, n_head=8, ctx=256, val=True, name="gpt_v"),
})
R3_SUITE = ["softmax_rot_v", "mlp_fmnist_v", "cnn_cifar_v", "charlm_v"]
R3_TRANSFER_SUITE = ["resnet18_cifar_v", "gpt_v", "mlp_fmnist_long_v", "cnn_cifar_bs512_v", "softmax_rot_long_v"]


def _renamed(task, name):
    task.name = name
    return task


def make_task(name, **overrides):
    task = TASKS[name]()
    for k, v in overrides.items():
        setattr(task, k, v)
    return task
