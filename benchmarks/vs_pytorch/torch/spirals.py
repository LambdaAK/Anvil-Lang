"""Two interleaved spirals, a 2-64-64-2 tanh network, Adam (the same as anvil/spirals.anvil): 4,000 steps
of almost no arithmetic, so the time is nearly all the cost of a training step itself."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, accuracy, batches, he, options, prepare

args = options(EPOCHS=200, CHUNK=20)
n, turns = 1000, 1.5
t = torch.linspace(0.3, 2 * math.pi * turns, n)
c = torch.arange(2).float()[:, None]
angle = t[None, :] + c * math.pi
r = (t / (2 * math.pi * turns))[None, :]
points = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=-1) + 0.03 * torch.randn(2, n, 2)
X = points.reshape(2 * n, 2).to(args.dev)
Y = (torch.arange(2 * n) >= n).long().to(args.dev)

net = prepare(he(nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))), args)
opt = torch.optim.Adam(net.parameters(), lr=0.01)
clock = Clock(args.dev)

for chunk in range(args.EPOCHS // args.CHUNK):
    clock.start()
    for epoch in range(args.CHUNK):
        for i in batches(len(X), 100, args.dev):
            loss = F.cross_entropy(net(X[i]), Y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    clock.stop()
print(f"@metric {accuracy(net, X, Y):.4f}")
