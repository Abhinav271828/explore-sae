import torch
import torch.nn.functional as F
from torch import nn, optim
import time
from pathlib import Path
import os
import matplotlib.pyplot as plt
from hparams import device

class AutoEncoder(nn.Module):
    def __init__(self, d_model, d_latent):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_latent, bias=True)
        self.act = nn.ReLU()

        self.ff2 = nn.Linear(d_latent, d_model, bias=False)
        self.dec_bias = nn.Parameter(torch.randn((d_model,)))

    def forward(self, x):
        latent = self.act(self.ff1(x - self.dec_bias))
        output = self.ff2(latent) + self.dec_bias
        return output, latent

d_model = 128
d_latent = 128
sae = AutoEncoder(d_model, d_latent).to(device)

if __name__ == '__main__':
    run_name = 'grok_1716823448'
    layer = 'embeddings'
    ckpt = 'final'
    data = torch.load(f'activations/{run_name}/{ckpt}_{layer}.pth', map_location=device)
    
    α = 1e-6
    opt = optim.Adam(sae.parameters(), lr=1e-3)
    stopping_thresh = -1

    root = Path('sae')
    (root/run_name).mkdir(parents=True,exist_ok=True)

    losses = []

    for epoch in range(10000):
        output, latent = sae(data)
        mse_loss = F.mse_loss(output, data)
        loss = mse_loss + α * (1/d_latent) * latent.norm()

        losses.append(loss.item())

        if epoch%100 == 0: print(f"{epoch}_{loss.item():.4f}")
    
        loss.backward()
        opt.step()
        opt.zero_grad()

        if loss.item() < stopping_thresh:
            break

    save_dict = {
        'model': sae.state_dict(),
        'optimizer': opt.state_dict(),
        'alpha': α,
        'loss': loss,
        'losses': losses,
        'epoch': epoch,
    }

    torch.save(save_dict, root/run_name/f"{ckpt}_{layer}_{d_latent}_{α}.pth")
    print(f"Saved model to {root/run_name/f'{ckpt}_{layer}_{d_latent}_{α}.pth'}")

    plt.plot(losses)
    plt.yscale('log')
    plt.show()
