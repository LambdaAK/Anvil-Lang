"""What the PyTorch programs share: options, data loading, Anvil's initializations, and the timing
protocol the runner reads (the same lines the Anvil programs print):

    @epoch <seconds>     the training time of one epoch (or one chunk of steps), GPU work included
    @metric <value>      the result: test accuracy, or a loss

Every program is written the way a PyTorch user would write it for speed: the whole dataset is a
tensor on the device, batches are gathered with a random permutation (no DataLoader), the loss is
never read back inside the loop, and `optimizer.zero_grad(set_to_none=True)`.
"""
import argparse
import gzip
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
MNIST = os.path.join(ROOT, "examples", "data")
DATA = os.path.join(os.path.dirname(HERE), "data")


def options(**consts):
    """--device cpu|mps, --compile, --threads N, and the program's constants (--EPOCHS 1 ...)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (default: PyTorch's choice)")
    ap.add_argument("--seed", type=int, default=0)
    for name, value in consts.items():
        ap.add_argument(f"--{name}", type=type(value), default=value)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.dev = torch.device(args.device)
    return args


def idx(name):
    with gzip.open(os.path.join(MNIST, name), "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def mnist(dev, shape=(-1, 784)):
    """MNIST as float tensors in [0, 1] and int64 labels, on the device."""
    x = torch.tensor(idx("train-images-idx3-ubyte.gz"), dtype=torch.float32).div_(255).reshape(shape)
    y = torch.tensor(idx("train-labels-idx1-ubyte.gz"), dtype=torch.int64)
    xt = torch.tensor(idx("t10k-images-idx3-ubyte.gz"), dtype=torch.float32).div_(255).reshape(shape)
    yt = torch.tensor(idx("t10k-labels-idx1-ubyte.gz"), dtype=torch.int64)
    return x.to(dev), y.to(dev), xt.to(dev), yt.to(dev)


def text(dev):
    with open(os.path.join(DATA, "readme.txt"), "rb") as f:
        return torch.tensor(np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)).to(dev)


def he(module):
    """Anvil's initialization: weights ~ normal(0, sqrt(2 / fan_in)) (He), biases zero."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / m.in_features))
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan_in))
            nn.init.zeros_(m.bias)
    return module


def prepare(model, args):
    model = model.to(args.dev)
    return torch.compile(model) if args.compile else model


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


class Clock:
    """Times one epoch: start() ... stop() prints `@epoch`, after the GPU has finished."""

    def __init__(self, dev):
        self.dev = dev

    def start(self):
        sync(self.dev)
        self.t = time.perf_counter()

    def stop(self):
        sync(self.dev)
        print(f"@epoch {time.perf_counter() - self.t:.5f}", flush=True)


def batches(n, size, dev):
    """Shuffled batch indices, dropping the last partial batch (as Anvil's `batches` does)."""
    perm = torch.randperm(n, device=dev)
    for k in range(n // size):
        yield perm[k * size:(k + 1) * size]


@torch.no_grad()
def accuracy(model, x, y, chunk=10_000):
    right = 0
    for k in range(0, len(x), chunk):
        right += (model(x[k:k + chunk]).argmax(1) == y[k:k + chunk]).sum().item()
    return right / len(x)
