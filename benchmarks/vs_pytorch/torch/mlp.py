"""A two-layer perceptron on MNIST (784-128-10), SGD (the same as anvil/mlp.anvil)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, accuracy, batches, he, mnist, options, prepare

args = options(EPOCHS=5, BATCH=64)
X, Y, X_test, Y_test = mnist(args.dev)
net = prepare(he(nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))), args)
opt = torch.optim.SGD(net.parameters(), lr=0.1)
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
