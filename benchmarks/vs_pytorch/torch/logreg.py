"""Softmax regression on MNIST: one 784×10 linear layer, SGD (the same as anvil/logreg.anvil)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, accuracy, batches, he, mnist, options, prepare

args = options(EPOCHS=5, BATCH=100)
X, Y, X_test, Y_test = mnist(args.dev)
net = prepare(he(nn.Linear(784, 10)), args)
opt = torch.optim.SGD(net.parameters(), lr=0.2)
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
