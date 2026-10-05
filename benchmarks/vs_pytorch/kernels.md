| operation | Anvil (anvil.function) | PyTorch eager | torch.compile | PyTorch MPS | Anvil vs eager | Anvil vs compile |
|---|---|---|---|---|---|---|
| matmul 1024×1024×1024 | 1.011 ms | 1.000 ms | 1.007 ms | 0.2223 ms | 0.99× | 1.00× |
| linear + GELU 256×1024→1024 | 0.5367 ms | 0.5511 ms | 0.6900 ms | 0.0914 ms | 1.03× | 1.29× |
| elementwise chain, 4M elements | 0.6081 ms | 5.108 ms | 0.8108 ms | 0.8961 ms | 8.40× | 1.33× |
| softmax 4096×1024 | 1.069 ms | 1.251 ms | 1.285 ms | 0.1023 ms | 1.17× | 1.20× |
| layer norm 4096×1024 | 0.7230 ms | 1.291 ms | 1.146 ms | 0.3830 ms | 1.79× | 1.59× |
| causal attention 16×256×64 | 1.230 ms | 0.4341 ms | 0.4799 ms | 0.0515 ms | 0.35× | 0.39× |
| conv 5×5 1→32 + ReLU + pool, 256×28×28 | 1.660 ms | 8.334 ms | 6.947 ms | 0.4310 ms | 5.02× | 4.19× |
| MLP 784-128-10, batch 1 (latency) | 0.0148 ms | 0.0057 ms | 0.0162 ms | 0.0422 ms | 0.38× | 1.09× |
| MLP 784-128-10, batch 1024 | 0.2716 ms | 0.2865 ms | 0.3433 ms | 0.0723 ms | 1.05× | 1.26× |

First call (compiling included for Anvil and torch.compile):

| operation | Anvil | PyTorch eager | torch.compile | PyTorch MPS |
|---|---|---|---|---|
| matmul 1024×1024×1024 | 0.00 s | 1.1 ms | 0.62 s | 40.5 ms |
| linear + GELU 256×1024→1024 | 0.19 s | 0.8 ms | 0.54 s | 477.7 ms |
| elementwise chain, 4M elements | 0.18 s | 8.0 ms | 0.04 s | 2.5 ms |
| softmax 4096×1024 | 0.18 s | 1.3 ms | 0.03 s | 6.3 ms |
| layer norm 4096×1024 | 0.18 s | 2.8 ms | 0.03 s | 2.0 ms |
| causal attention 16×256×64 | 0.19 s | 1.0 ms | 0.03 s | 0.7 ms |
| conv 5×5 1→32 + ReLU + pool, 256×28×28 | 0.19 s | 14.3 ms | 0.04 s | 26.9 ms |
| MLP 784-128-10, batch 1 (latency) | 0.18 s | 0.1 ms | 0.03 s | 0.9 ms |
| MLP 784-128-10, batch 1024 | 0.18 s | 0.7 ms | 0.03 s | 7.7 ms |
