from hparams import lr, weight_decay, num_epochs, save_models, save_every, stopping_thresh, device
from model import torch_model as model
from data import train, test
from utils import full_loss, lines

import numpy as np
import torch
import torch.optim as optim

import time
import os
from pathlib import Path

model.to(device)
train = train.to(device)
test = test.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
run_name = f"grok_{int(time.time())}"
print(f'Run name {run_name}')

root = Path('save')
(root/run_name).mkdir(parents=True,exist_ok=True)
save_dict = {'model':model.state_dict(), 'train_data': train, 'test_data':test}
torch.save(save_dict, root/run_name/'init.pth')

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = full_loss(model, train)
    test_loss = full_loss(model, test)

    if train_loss < 0:
        print(model(train)[:, -1])

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if epoch%100 == 0: print(f"{epoch}_{train_loss.item():.4f}_{test_loss.item():.4f}")

    train_loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if test_loss.item() < stopping_thresh:
        break

    if (save_models) and (epoch%save_every == 0):
        if test_loss.item() < stopping_thresh:
            break
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epoch': epoch,
        }
        torch.save(save_dict, root/run_name/f"{epoch}.pth")
        print(f"Saved model to {root/run_name/f'{epoch}.pth'}")

if not save_models:
    os.mkdir(root/run_name)

save_dict = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_losses': train_losses,
    'test_losses': test_losses,
    'epoch': epoch,
}

torch.save(save_dict, root/run_name/f"final.pth")
print(f"Saved model to {root/run_name/f'final.pth'}")

lines([train_losses, test_losses], labels=['train', 'test'], log_y=True)