"""A small GPT on a copy of the README: 2 blocks, 4 heads, width 64, context 64, Adam (the same as
anvil/gpt.anvil; the causal attention is F.scaled_dot_product_attention). The metric is bits per byte
over the last 100 steps."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, he, options, prepare, text


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.Wq = nn.Parameter(torch.randn(d, d) * math.sqrt(1 / d))
        self.Wk = nn.Parameter(torch.randn(d, d) * math.sqrt(1 / d))
        self.Wv = nn.Parameter(torch.randn(d, d) * math.sqrt(1 / d))
        self.Wo = nn.Parameter(torch.randn(d, d) * math.sqrt(0.5 / d))
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        h = F.layer_norm(x, (d,))
        q, k, v = ((h @ w).view(b, t, self.heads, d // self.heads).transpose(1, 2) for w in (self.Wq, self.Wk, self.Wv))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + o.transpose(1, 2).reshape(b, t, d) @ self.Wo
        return x + self.down(F.gelu(self.up(F.layer_norm(x, (d,))), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, t, d, heads):
        super().__init__()
        self.E = nn.Parameter(torch.randn(256, d) * 0.5)
        self.P = nn.Parameter(torch.randn(t, d) * 0.1)
        self.blocks = nn.Sequential(Block(d, heads), Block(d, heads))
        self.out = nn.Linear(d, 256)

    def forward(self, x):                                 # x: [batch, T + 1] -> the mean loss
        h = self.blocks(self.E[x[:, :-1]] + self.P)
        logits = self.out(F.layer_norm(h, (h.shape[-1],)))
        return F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))


args = options(STEPS=1000, CHUNK=100, BATCH=16, T=64, D=64, HEADS=4)
data = text(args.dev)
offsets = torch.arange(args.T + 1, device=args.dev)
net = prepare(he(GPT(args.T, args.D, args.HEADS)), args)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
clock = Clock(args.dev)

for chunk in range(args.STEPS // args.CHUNK):
    clock.start()
    total = torch.zeros((), device=args.dev)
    for step in range(args.CHUNK):
        at = torch.randint(0, len(data) - args.T - 1, (args.BATCH,), device=args.dev)
        loss = net(data[at[:, None] + offsets])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.detach()
    clock.stop()
print(f"@metric {total.item() / args.CHUNK / math.log(2):.4f}")
