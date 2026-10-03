"""A wide perceptron on MNIST (784-1024-1024-10), Adam (the same as anvil/wide_mlp.anvil)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, accuracy, batches, he, mnist, options, prepare

args = options(EPOCHS=2, BATCH=128)
X, Y, X_test, Y_test = mnist(args.dev)
net = prepare(he(nn.Sequential(nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                               nn.Linear(1024, 10))), args)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
clock = Clock(args.dev)

for epoch in range(args.EPOCHS):
    clock.start()
    for i in batches(len(X), args.BATCH, args.dev):
        loss = F.cross_entropy(net(X[i]), Y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    clock.stop()
print(f"@metric {accuracy(net, X_test, Y_test):.4f}")
