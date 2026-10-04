#!/bin/sh
# The same projects with other random seeds: is a difference in results between Anvil and PyTorch
# larger than the difference between two seeds of either? (Results: build/seeds.txt)
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/benchmarks/.venv-torch/bin/python"
for p in gpt wide_mlp charrnn; do
  for s in 1 2 3; do
    e=$(../../bin/anvil run --seed $s anvil/$p.anvil | grep @metric | cut -d' ' -f2)
    t=$(cd torch && "$PY" $p.py --seed $s | grep @metric | cut -d' ' -f2)
    echo "$p seed $s: anvil $e torch $t"
  done
done
