# Experiments

Run with `python3 experiments/run.py` on an Apple M3 Max, 2026-10-07 (171 s in all; load average 7.8 at the start, 10.9 at the end). MNIST: 60,000 training images, accuracy on the 10,000 test images.

## Ablation: seconds per training epoch (and how much slower than with everything on)

| configuration | MLP 784-128-10 | MLP 784-1024-1024-10 | LeNet CNN |
|---|---|---|---|
| everything on | 0.084 s (1.0×) | 1.949 s (1.0×) | 0.85 s (1.0×) |
| one thread | 0.294 s (3.5×) | 10.348 s (5.3×) | 2.85 s (3.4×) |
| no IR optimizations (-O0) | 0.136 s (1.6×) | 2.094 s (1.1×) | 2.55 s (3.0×) |
| -O0, one thread | 0.316 s (3.8×) | 9.312 s (4.8×) | 7.95 s (9.4×) |

## Thread scaling: seconds per epoch

| threads | MLP 784-128-10 | speed-up | MLP 784-1024-1024-10 | speed-up |
|---|---|---|---|---|
| 1 | 0.296 s | 1.00× | 10.666 s | 1.00× |
| 2 | 0.179 s | 1.65× | 5.481 s | 1.95× |
| 4 | 0.112 s | 2.65× | 3.167 s | 3.37× |
| 8 | 0.080 s | 3.71× | 1.872 s | 5.70× |
| 12 | 0.123 s | 2.42× | 2.594 s | 4.11× |

## Optimizers and learning rates: test accuracy after 3 epochs

| optimizer | lr #1 | lr #2 | lr #3 | lr #4 | lr #5 |
|---|---|---|---|---|---|
| sgd | 91.20% (lr 0.01) | 93.40% (lr 0.03) | 96.20% (lr 0.1) | 97.21% (lr 0.3) | 95.93% (lr 1) |
| adam | 93.07% (lr 0.0001) | 95.56% (lr 0.0003) | 97.47% (lr 0.001) | 97.08% (lr 0.003) | 96.55% (lr 0.01) |
| rmsprop | 93.43% (lr 0.0001) | 95.93% (lr 0.0003) | 97.37% (lr 0.001) | 97.32% (lr 0.003) | 96.66% (lr 0.01) |

Best: adam, lr 0.001 at 97.47%.

## Width and depth: test accuracy after 3 epochs, and seconds per epoch (Adam, lr 1e-3)

| hidden width | 1 hidden layer | 2 hidden layers |
|---|---|---|
| 32 | 95.33%, 0.03 s/epoch | 95.71%, 0.04 s/epoch |
| 64 | 96.55%, 0.05 s/epoch | 97.10%, 0.07 s/epoch |
| 128 | 97.47%, 0.10 s/epoch | 97.50%, 0.13 s/epoch |
| 256 | 97.65%, 0.13 s/epoch | 97.28%, 0.19 s/epoch |
| 512 | 97.63%, 0.29 s/epoch | 97.70%, 0.55 s/epoch |
| 1024 | 97.81%, 0.81 s/epoch | 97.86%, 2.27 s/epoch |
