"""Get examples/sentiment.anvil's data ready: the pretrained model and SST-2, tokenized.

    python3 examples/sentiment/prepare.py

- The model is BAAI/bge-small-en-v1.5 from Hugging Face. If it is in the Hugging Face cache
  (~/.cache/huggingface/hub), examples/data/bge-small becomes a link to it. Otherwise this prints how
  to download it (127 MB).
- SST-2 (the Stanford Sentiment Treebank, as in GLUE) must be in examples/data/sst2: train.tsv and
  dev.tsv, from https://dl.fbaipublicfiles.com/glue/data/SST-2.zip (7.4 MB).

The sentences become token ids (BERT's WordPiece, tokenizer.py), padded with 0. Training sentences
are sorted into three files by length, at most 16, 32 and 64 tokens, so that short sentences (most of
them) are not padded to 64: training is about 3 times faster. The dev set is one file of 64.
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
sys.path.insert(0, HERE)
from tokenizer import Tokenizer  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"
BUCKETS = (16, 32, 64)


def find_model() -> str:
    link = os.path.join(DATA, "bge-small")
    if os.path.exists(os.path.join(link, "model.safetensors")):
        return link
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + MODEL.replace("/", "--"), "snapshots", "*", "model.safetensors")))
    if not snaps:
        sys.exit(f"{MODEL} is not in the Hugging Face cache. Download it with\n\n"
                 f"    huggingface-cli download {MODEL}\n\n"
                 f"or put model.safetensors and vocab.txt from https://huggingface.co/{MODEL} in {link}/")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(os.path.dirname(snaps[-1]), link)
    return link


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return [(r[0], int(r[1])) for r in rows[1:]]


def pad(seqs, width):
    out = np.zeros((len(seqs), width), dtype=np.int32)
    for k, s in enumerate(seqs):
        if len(s) > width:                      # too long: keep the start, and end with [SEP]
            s = s[:width - 1] + s[-1:]
        out[k, :len(s)] = s
    return out


def main():
    model = find_model()
    tok = Tokenizer(os.path.join(model, "vocab.txt"))
    sst = os.path.join(DATA, "sst2")
    if not os.path.exists(os.path.join(sst, "train.tsv")):
        sys.exit(f"SST-2 is missing: unzip train.tsv and dev.tsv from "
                 f"https://dl.fbaipublicfiles.com/glue/data/SST-2.zip into {sst}/")

    train = [(tok.encode(s), y) for s, y in read_tsv(os.path.join(sst, "train.tsv"))]
    lo = 0
    for hi in BUCKETS:
        part = [(s, y) for s, y in train if lo < len(s) <= hi or hi == BUCKETS[-1] and len(s) > hi]
        np.save(os.path.join(sst, f"train{hi}_ids.npy"), pad([s for s, _ in part], hi))
        np.save(os.path.join(sst, f"train{hi}_labels.npy"), np.array([y for _, y in part], dtype=np.int32))
        print(f"train, {lo + 1:2d}-{hi} tokens: {len(part):6,d} sentences")
        lo = hi
    dev = [(tok.encode(s), y) for s, y in read_tsv(os.path.join(sst, "dev.tsv"))]
    np.save(os.path.join(sst, "dev_ids.npy"), pad([s for s, _ in dev], BUCKETS[-1]))
    np.save(os.path.join(sst, "dev_labels.npy"), np.array([y for _, y in dev], dtype=np.int32))
    print(f"dev:                {len(dev):6,d} sentences")
    print(f"model: {os.path.realpath(model)}")


if __name__ == "__main__":
    main()
