"""Data the benchmarks share that is not already in examples/data: a fixed copy of the README (the
text models' corpus; the real README keeps changing) and a 120,000-image slice of EMNIST as .npy
files, which both Anvil (`npy`) and PyTorch load in a fraction of a second.

    python3 benchmarks/vs_pytorch/prepare.py
"""
import gzip
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
EMNIST = os.path.join(ROOT, "examples", "data", "emnist")
N_TRAIN, N_TEST = 120_000, 10_000


def idx(path, count):
    """The first `count` items of a gzipped IDX file, as uint8."""
    with gzip.open(path, "rb") as f:
        head = f.read(4)
        dims = [int.from_bytes(f.read(4), "big") for _ in range(head[3])]
        per = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        data = np.frombuffer(f.read(count * per), dtype=np.uint8)
    return data.reshape([count] + dims[1:])


def main():
    os.makedirs(DATA, exist_ok=True)
    text = os.path.join(DATA, "readme.txt")
    if not os.path.exists(text):
        shutil.copy(os.path.join(ROOT, "README.md"), text)
    out = os.path.join(DATA, "emnist_test_y.npy")
    if not os.path.exists(out):
        if not os.path.exists(EMNIST):
            raise SystemExit("EMNIST is not in examples/data/emnist (see examples/letters.anvil)")
        for split, n in (("train", N_TRAIN), ("test", N_TEST)):
            x = idx(os.path.join(EMNIST, f"emnist-byclass-{split}-images-idx3-ubyte.gz"), n)
            y = idx(os.path.join(EMNIST, f"emnist-byclass-{split}-labels-idx1-ubyte.gz"), n)
            np.save(os.path.join(DATA, f"emnist_{split}_x.npy"), np.ascontiguousarray(x))
            np.save(os.path.join(DATA, f"emnist_{split}_y.npy"), y.astype(np.int32))
    print("data ready in", DATA)


if __name__ == "__main__":
    main()
