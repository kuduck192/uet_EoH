"""Download Fashion-MNIST, CIFAR-10 (train and test splits) and TinyShakespeare and
store them as compact tensor files ($NNOPT_DATA/{fmnist,fmnist_test,cifar10,
cifar10_test,shakespeare}.pt) that tasks.py loads straight onto the GPU.

    python prepare_data.py            # writes ./data/*.pt
    NNOPT_DATA=/dev/shm/nnopt python prepare_data.py

The image datasets come from the Hugging Face Hub (the torchvision mirrors are
unreachable from the training server); set HF_HOME to choose the cache.
"""

import os
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import DATA_DIR

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def hf_images(repo, image_col, label_col="label", split="train"):
    """One split of an HF image dataset as (uint8 array (N, H, W[, C]), int64 labels)."""
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    x = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[image_col]])
    y = np.asarray(ds[label_col], dtype=np.int64)
    return x, y


def main():
    raw = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw, exist_ok=True)

    out = os.path.join(DATA_DIR, "fmnist.pt")
    if not os.path.exists(out):
        x, y = hf_images("zalando-datasets/fashion_mnist", "image")
        torch.save({"x": torch.from_numpy(x.reshape(-1, 784)).contiguous(), "y": torch.from_numpy(y)}, out)
        print("wrote", out, tuple(x.shape))

    out = os.path.join(DATA_DIR, "cifar10.pt")
    if not os.path.exists(out):
        x, y = hf_images("uoft-cs/cifar10", "img")
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # (N, 3, 32, 32) uint8
        torch.save({"x": x, "y": torch.from_numpy(y)}, out)
        print("wrote", out, tuple(x.shape))

    # Test splits, used as validation data by the R3 tasks.
    out = os.path.join(DATA_DIR, "fmnist_test.pt")
    if not os.path.exists(out):
        x, y = hf_images("zalando-datasets/fashion_mnist", "image", split="test")
        torch.save({"x": torch.from_numpy(x.reshape(-1, 784)).contiguous(), "y": torch.from_numpy(y)}, out)
        print("wrote", out, tuple(x.shape))

    out = os.path.join(DATA_DIR, "cifar10_test.pt")
    if not os.path.exists(out):
        x, y = hf_images("uoft-cs/cifar10", "img", split="test")
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        torch.save({"x": x, "y": torch.from_numpy(y)}, out)
        print("wrote", out, tuple(x.shape))

    out = os.path.join(DATA_DIR, "shakespeare.pt")
    if not os.path.exists(out):
        path = os.path.join(raw, "tinyshakespeare.txt")
        if not os.path.exists(path):
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        vocab = sorted(set(text))
        lookup = {c: i for i, c in enumerate(vocab)}
        tokens = torch.from_numpy(np.array([lookup[c] for c in text], dtype=np.uint8))
        torch.save({"tokens": tokens, "vocab_size": len(vocab), "vocab": "".join(vocab)}, out)
        print("wrote", out, len(tokens), "tokens, vocab", len(vocab))


if __name__ == "__main__":
    main()
