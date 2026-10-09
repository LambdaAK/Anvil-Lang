"""examples/sentiment.anvil in PyTorch: the same model (reference.py), data, batches, schedule and
dropout, to compare accuracy and speed.

    benchmarks/.venv-torch/bin/python examples/sentiment/torch_finetune.py --time          time steps
    benchmarks/.venv-torch/bin/python examples/sentiment/torch_finetune.py                 fine-tune fully
    ... --device mps        on the Apple GPU          ... --threads 8        CPU threads
"""
import argparse
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
sys.path.insert(0, HERE)
from reference import Sentiment, read_safetensors  # noqa: E402

EPOCHS, BATCH, LR, DROP = 2, 32, 5e-5, 0.1


def load(name):
    return np.load(os.path.join(DATA, "sst2", name))


def warmup_cosine(step, total, peak, warmup):            # as in Anvil's prelude
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--time", action="store_true", help="time 30 steps of each length instead of training")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    net = Sentiment(read_safetensors(os.path.join(DATA, "bge-small", "model.safetensors"))).to(dev)
    step_fn = torch.compile(net) if a.compile else net
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buckets = [(load(f"train{n}_ids.npy"), load(f"train{n}_labels.npy")) for n in (16, 32, 64)]
    per_epoch = sum(len(y) // BATCH for _, y in buckets)
    total = EPOCHS * per_epoch
    sync = (lambda: torch.mps.synchronize()) if a.device == "mps" else (lambda: None)

    def train_step(x, y, step):
        for g in opt.param_groups:
            g["lr"] = warmup_cosine(step, total, LR, total * 6 // 100)
        loss = F.cross_entropy(step_fn(x, DROP), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return loss

    print(f"PyTorch {torch.__version__} on {a.device}, {torch.get_num_threads()} threads")
    net.train()
    if a.time:
        for (X, Y), n in zip(buckets, (16, 32, 64)):
            times = []
            for k in range(35):
                x = torch.from_numpy(X[k * BATCH:(k + 1) * BATCH].astype(np.int64)).to(dev)
                y = torch.from_numpy(Y[k * BATCH:(k + 1) * BATCH].astype(np.int64)).to(dev)
                sync()
                t0 = time.perf_counter()
                train_step(x, y, k)
                sync()
                times.append(time.perf_counter() - t0)
            print(f"  {n} tokens: {statistics.median(times[5:]) * 1000:.1f} ms per step (median of 30)")
        return

    rng = np.random.default_rng(0)
    step, start = 0, time.perf_counter()
    for epoch in range(EPOCHS):
        for X, Y in buckets:
            order = rng.permutation(len(Y))
            for k in range(len(Y) // BATCH):
                pick = order[k * BATCH:(k + 1) * BATCH]
                x = torch.from_numpy(X[pick].astype(np.int64)).to(dev)
                y = torch.from_numpy(Y[pick].astype(np.int64)).to(dev)
                train_step(x, y, step)
                step += 1
                if step % 400 == 0:
                    sync()
                    print(f"step {step:5d} of {total}   ({time.perf_counter() - start:.0f} s)", flush=True)
        net.eval()
        with torch.no_grad():
            X, Y = load("dev_ids.npy"), load("dev_labels.npy")
            pred = net(torch.from_numpy(X.astype(np.int64)).to(dev)).argmax(1).cpu().numpy()
        net.train()
        print(f"epoch {epoch + 1}   dev accuracy {(pred == Y).mean():.2%}   ({time.perf_counter() - start:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
