"""
In this file we provide basic functions to train and evaluate.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_for_one_epoch(dataset, loss, net, optimizer, device='cpu', batch_size=32, plot: bool = False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    net.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        x = x.to(device=device)
        y = y.to(device=device)
        y_hat = net(x)
        loss_y_yhat = loss(y_hat, y)
        loss_y_yhat.backward()
        optimizer.step()
        losses.append(loss_y_yhat.detach().item())
    if plot:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(y[0, 0].cpu())
        ax2.imshow(y_hat[0, 0].detach().cpu())
        plt.show()
    return np.mean(losses)

def test(dataset, loss, net, device='cpu', batch_size=32, plot: bool = False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
            y_hat = net(x)
            loss_y_yhat = loss(y_hat, y)
            losses.append(loss_y_yhat.detach().item())
    if plot:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(y[0, 0].cpu())
        ax2.imshow(y_hat[0, 0].detach().cpu())
        plt.show()
    return np.mean(losses)