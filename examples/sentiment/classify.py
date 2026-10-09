"""Read the sentiment of sentences with the fine-tuned model (examples/sentiment.weights).

    python3 examples/sentiment/classify.py "a gorgeous, witty, seductive movie" "the plot is a mess"
    python3 examples/sentiment/classify.py            (then type sentences, one per line)

The sentences are tokenized here, then classify.anvil runs the model on them, 16 at a time (it is
compiled once, and kept in the build cache after that).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "examples", "data")
sys.path.insert(0, HERE)
from tokenizer import Tokenizer  # noqa: E402

N, T = 16, 64


def classify(sentences: list[str], tok: Tokenizer) -> np.ndarray:
    out = []
    for i in range(0, len(sentences), N):
        part = sentences[i:i + N]
        ids = np.zeros((N, T), dtype=np.int32)
        for k, s in enumerate(part):
            seq = tok.encode(s)
            seq = seq[:T - 1] + seq[-1:] if len(seq) > T else seq
            ids[k, :len(seq)] = seq
        np.save(os.path.join(DATA, "sst2", "classify_ids.npy"), ids)
        subprocess.run([os.path.join(ROOT, "bin", "anvil"), "run", os.path.join(HERE, "classify.anvil")], check=True)
        out.append(np.load(os.path.join(DATA, "sst2", "classify_out.npy"))[:len(part)])
    return np.concatenate(out)


def show(sentences, probs):
    for s, p in zip(sentences, probs):
        good = p[1] >= 0.5
        mood = "\033[32mpositive\033[0m" if good else "\033[31mnegative\033[0m"
        print(f"  {mood} {max(p) * 100:5.1f}%   {s}")


def main():
    tok = Tokenizer(os.path.join(DATA, "bge-small", "vocab.txt"))
    if len(sys.argv) > 1:
        sentences = sys.argv[1:]
        show(sentences, classify(sentences, tok))
        return
    print("type a sentence (an empty line ends):")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        show([line], classify([line], tok))


if __name__ == "__main__":
    main()
