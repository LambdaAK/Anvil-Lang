"""Tasks for examples/synth.anvil: random Pipes programs (pipes.py), each with eight lists run
through it.

    python3 examples/synth/prepare.py

examples/data/synth/train.npy holds 1,800,000 tasks as token ids (one byte each): four examples
(input and output) and the program, 80 tokens each. A tenth of all programs (by a hash of their text) never appear in
training: the 2,000 test tasks use only those, so the model is tested on programs it has never seen.
test.jsonl keeps each test task's program and all eight examples (four shown, four hidden).
"""
import hashlib
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "synth")
sys.path.insert(0, HERE)
import pipes as P  # noqa: E402

N_TRAIN, N_TEST = 1_800_000, 2_000


def held_out(program) -> bool:
    return hashlib.sha256(P.show(program).encode()).digest()[0] % 10 == 0


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)
    train = np.zeros((N_TRAIN, P.SEQ), dtype=np.uint8)          # (the 227 tokens fit in a byte)
    n, test, programs = 0, [], set()
    while n < N_TRAIN or len(test) < N_TEST:
        program, pairs = P.random_task(rng)
        if held_out(program):
            if len(test) < N_TEST:
                test.append((program, pairs))
        elif n < N_TRAIN:
            train[n] = P.encode_spec(pairs[:P.N_SHOWN]) + P.encode_program(program)
            programs.add(P.show(program))
            n += 1
    np.save(os.path.join(OUT, "train.npy"), train)
    np.save(os.path.join(OUT, "test.npy"), np.array([P.encode_spec(pairs[:P.N_SHOWN]) + P.encode_program(program)
                                                       for program, pairs in test], dtype=np.int32))
    with open(os.path.join(OUT, "test.jsonl"), "w") as f:
        for program, pairs in test:
            f.write(json.dumps({"program": P.show(program), "examples": pairs}) + "\n")
    print(f"{n:,} training tasks ({len(programs):,} different programs), {len(test):,} test tasks "
          f"(programs never seen in training), {len(P.VOCAB)} tokens, {P.SEQ} per task")


if __name__ == "__main__":
    main()
