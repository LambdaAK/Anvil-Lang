"""Does the Anvil clone compute what the model computes? Runs clone_check.anvil (the first 16 dev
sentences through examples/bge_small.anvil) and the same sentences through reference.py in PyTorch,
and compares every token's final vector.

    benchmarks/.venv-torch/bin/python examples/sentiment/check_clone.py
"""
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "examples", "data")
sys.path.insert(0, HERE)
from reference import Bert, read_safetensors  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402


def main():
    subprocess.run([os.path.join(ROOT, "bin", "anvil"), "run", os.path.join(HERE, "clone_check.anvil")], check=True)
    ours = np.load(os.path.join(DATA, "sst2", "clone_hidden.npy"))
    ids = np.load(os.path.join(DATA, "sst2", "dev_ids.npy"))[:16]
    torch.set_num_threads(8)
    bert = Bert(read_safetensors(os.path.join(DATA, "bge-small", "model.safetensors"))).eval()
    with torch.no_grad():
        theirs = bert(torch.from_numpy(ids.astype(np.int64))).numpy()

    real = ids != 0                                     # padding's vectors are not used by anything
    diff = np.abs(ours - theirs)[real]
    cls_a, cls_b = ours[:, 0], theirs[:, 0]
    cos = (cls_a * cls_b).sum(1) / np.linalg.norm(cls_a, axis=1) / np.linalg.norm(cls_b, axis=1)
    print(f"{int(real.sum())} tokens of 16 sentences, after 12 layers:")
    print(f"  largest difference  {diff.max():.2e}   (values up to {np.abs(theirs[real]).max():.1f})")
    print(f"  mean difference     {diff.mean():.2e}")
    print(f"  [CLS] cosine        {cos.min():.8f} (lowest of the 16)")

    tok = Tokenizer(os.path.join(DATA, "bge-small", "vocab.txt"))
    print(f'\n  e.g. "{tok.decode(ids[0][1:real[0].sum() - 1])}"')
    ok = diff.max() < 1e-3 and cos.min() > 0.99999
    print("\nthe clone matches" if ok else "\nTHE CLONE DIFFERS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
