"""A GRU language model on a copy of the README: 32 steps of backpropagation through time, Adam (the
same as anvil/charrnn.anvil, but with PyTorch's own nn.GRU, which puts the reset gate after the hidden
state's linear map instead of before it). The metric is bits per byte over the last 100 steps."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, he, options, prepare, text


class CharRNN(nn.Module):
    def __init__(self, embed, hidden):
        super().__init__()
        self.E = nn.Embedding(256, embed)                 # normal(0, 1), as in Anvil
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 256)
        for name, w in self.gru.named_parameters():       # Anvil's initialization
            if name.startswith("weight"):
                nn.init.normal_(w, 0.0, math.sqrt(2.0 / w.shape[1]))
            else:
                nn.init.zeros_(w)

    def forward(self, x):                                 # x: [batch, seq + 1] -> the mean loss
        h, _ = self.gru(self.E(x[:, :-1]))
        return F.cross_entropy(self.out(h).reshape(-1, 256), x[:, 1:].reshape(-1))


args = options(STEPS=1000, CHUNK=100, BATCH=32, SEQ=32, EMBED=32, HIDDEN=128)
data = text(args.dev)
offsets = torch.arange(args.SEQ + 1, device=args.dev)
net = prepare(he(CharRNN(args.EMBED, args.HIDDEN)), args)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
clock = Clock(args.dev)

for chunk in range(args.STEPS // args.CHUNK):
    clock.start()
    total = torch.zeros((), device=args.dev)
    for step in range(args.CHUNK):
        at = torch.randint(0, len(data) - args.SEQ - 1, (args.BATCH,), device=args.dev)
        loss = net(data[at[:, None] + offsets])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.detach()
    clock.stop()
print(f"@metric {total.item() / args.CHUNK / math.log(2):.4f}")
