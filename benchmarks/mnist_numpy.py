"""Baseline: the same MLP as examples/mnist.anvil, trained with NumPy.

784-128-10, ReLU, He init, cross-entropy, batch 64, SGD lr=0.1, reshuffled every epoch.
Run:  python3 benchmarks/mnist_numpy.py
"""
import gzip
import os
import time

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data")


def idx(name):
    with gzip.open(os.path.join(DATA, name), "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


X = idx("train-images-idx3-ubyte.gz").reshape(60000, 784).astype(np.float32) / 255
Y = idx("train-labels-idx1-ubyte.gz").astype(np.int64)
Xt = idx("t10k-images-idx3-ubyte.gz").reshape(10000, 784).astype(np.float32) / 255
Yt = idx("t10k-labels-idx1-ubyte.gz").astype(np.int64)

rng = np.random.default_rng(0)
W1 = (rng.standard_normal((784, 128)) * np.sqrt(2 / 784)).astype(np.float32)
b1 = np.zeros(128, np.float32)
W2 = (rng.standard_normal((128, 10)) * np.sqrt(2 / 128)).astype(np.float32)
b2 = np.zeros(10, np.float32)
B, lr = 64, np.float32(0.1)

start = time.perf_counter()
train_time = 0.0                       # without the test-set evaluations
for epoch in range(5):
    t_epoch = time.perf_counter()
    perm = rng.permutation(60000)
    for s in range(60000 // B):
        idx_ = perm[s * B:(s + 1) * B]
        x, y = X[idx_], Y[idx_]
        z1 = x @ W1 + b1
        h = np.maximum(z1, 0)
        z2 = h @ W2 + b2
        z2 = z2 - z2.max(axis=1, keepdims=True)
        e = np.exp(z2)
        p = e / e.sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(B), y]).mean()
        dz2 = p
        dz2[np.arange(B), y] -= 1
        dz2 /= B
        dW2 = h.T @ dz2
        db2 = dz2.sum(0)
        dh = dz2 @ W2.T
        dz1 = dh * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    train_time += time.perf_counter() - t_epoch
    acc = ((np.maximum(Xt @ W1 + b1, 0) @ W2 + b2).argmax(1) == Yt).mean()
    print(f"epoch {epoch + 1}   loss {loss:.4f}   test accuracy {acc:.2%}   ({time.perf_counter() - start:.3f}s)")
print(f"training: {train_time / 5:.4f} s per epoch")
