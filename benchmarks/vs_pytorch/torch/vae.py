"""A variational autoencoder on MNIST (784-256-16-256-784), Adam (the same as anvil/vae.anvil). The
metric is the last epoch's mean loss per image (reconstruction + KL)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Clock, batches, he, mnist, options, prepare


class VAE(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.enc = nn.Linear(784, 256)
        self.mu = nn.Linear(256, z)
        self.logvar = nn.Linear(256, z)
        self.dec1 = nn.Linear(z, 256)
        self.dec2 = nn.Linear(256, 784)

    def forward(self, x):                       # the loss: reconstruction (per image) + KL
        h = F.relu(self.enc(x))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon = F.binary_cross_entropy(torch.sigmoid(self.dec2(F.relu(self.dec1(z)))), x) * 784
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu * mu - torch.exp(logvar), dim=1))
        return recon + kl


args = options(EPOCHS=3, BATCH=100, Z=16)
X, _, _, _ = mnist(args.dev)
net = prepare(he(VAE(args.Z)), args)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
clock = Clock(args.dev)

for epoch in range(args.EPOCHS):
    clock.start()
    total = torch.zeros((), device=args.dev)
    for i in batches(len(X), args.BATCH, args.dev):
        loss = net(X[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.detach()
    clock.stop()
print(f"@metric {total.item() / (len(X) // args.BATCH):.4f}")
