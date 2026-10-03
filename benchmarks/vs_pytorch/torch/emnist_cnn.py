"""The letter network of bin/draw --text (conv 32, conv 64, linear 256, dropout 0.3, 62 classes) on
120,000 EMNIST characters, Adam (the same as anvil/emnist_cnn.anvil)."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, Clock, accuracy, batches, he, options, prepare


class LetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 5)
        self.c2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 62)

    def forward(self, x):
        h = F.max_pool2d(F.relu(self.c1(x)), 2)
        h = F.max_pool2d(F.relu(self.c2(h)), 2).flatten(1)
        return self.fc2(F.dropout(F.relu(self.fc1(h)), 0.3, self.training))


def load(name):
    return np.load(os.path.join(DATA, name))


args = options(EPOCHS=1, BATCH=128)
X = torch.tensor(load("emnist_train_x.npy"), dtype=torch.float32).div_(255).reshape(-1, 1, 28, 28).to(args.dev)
Y = torch.tensor(load("emnist_train_y.npy"), dtype=torch.int64).to(args.dev)
X_test = torch.tensor(load("emnist_test_x.npy"), dtype=torch.float32).div_(255).reshape(-1, 1, 28, 28).to(args.dev)
Y_test = torch.tensor(load("emnist_test_y.npy"), dtype=torch.int64).to(args.dev)
model = he(LetterNet())
net = prepare(model, args)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
clock = Clock(args.dev)

for epoch in range(args.EPOCHS):
    model.train()
    clock.start()
    for i in batches(len(X), args.BATCH, args.dev):
        loss = F.cross_entropy(net(X[i]), Y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    clock.stop()
model.eval()
print(f"@metric {accuracy(net, X_test, Y_test, chunk=1000):.4f}")
