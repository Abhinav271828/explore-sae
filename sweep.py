import torch
from pathlib import Path
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder, train_sae
from hparams import device

d_model = 128
d_latent = 128

run_name = 'grok_1716823448'
layer = 'embeddings'
ckpt = 'final'
data = torch.load(f'activations/{run_name}/{ckpt}_{layer}.pth', map_location=device)

alphas = []
num_runs = 25
losses = torch.zeros(18, 2, num_runs)
i = 0

for exp in [-3, -2, -1, 0, 1, 2]:
    for c in [1, 2, 5]:
        α = eval(f'{c}e{exp}')
        print(f"Running with α = {α}")
    
        root = Path('sae')
        (root/run_name).mkdir(parents=True, exist_ok=True)
        save_path = root/run_name/f"{ckpt}_{layer}_{d_latent}_{c}e{exp}.pth"

        for t in range(num_runs):
            print(f"Run {t+1}/{num_runs}")
            cur_mse_losses, cur_reg_losses, _ = train_sae(AutoEncoder(d_model, d_latent).to(device), data, α, save_path)
            losses[i, 0, t] = cur_mse_losses[-1]
            losses[i, 1, t] = cur_reg_losses[-1]

        alphas.append(α)
        print("Average MSE loss:", losses[i, 0].mean().item())
        print("Average Reg loss:", losses[i, 1].mean().item())
        i += 1

torch.save(losses, f'losses-{d_latent}-{num_runs}.pt')
plt.figure()
plt.errorbar(alphas, losses.mean(dim=2)[:, 0], yerr=losses.std(dim=2)[0], label='MSE')
plt.errorbar(alphas, losses.mean(dim=2)[:, 1], yerr=losses.std(dim=2)[1], label='Reg')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()