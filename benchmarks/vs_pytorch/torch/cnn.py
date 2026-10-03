"""A LeNet-style CNN on MNIST (conv 8, conv 16, linear), Adam (the same as anvil/cnn.anvil)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, accuracy, batches, he, mnist, options, prepare


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 5)
        self.c2 = nn.Conv2d(8, 16, 5)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


args = options(EPOCHS=2, BATCH=50)
X, Y, X_test, Y_test = mnist(args.dev, shape=(-1, 1, 28, 28))
net = prepare(he(LeNet()), args)
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
clock = Clock(args.dev)

for epoch in range(args.EPOCHS):
    clock.start()
    for i in batches(len(X), args.BATCH, args.dev):
        loss = F.cross_entropy(net(X[i]), Y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    clock.stop()
print(f"@metric {accuracy(net, X_test, Y_test, chunk=1000):.4f}")
