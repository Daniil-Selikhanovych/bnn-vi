import time
from IPython import display
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm


class ProgressPlotter:
    def __init__(self, losses, update_per=5):
        self.losses = losses
        self.update_per = update_per
        
    def start(self):
        self.time = time.time()
        
    def update(self, num_iter):
        if num_iter % self.update_per != 0:
            return
        old_time = self.time
        self.time = time.time()
        iter_time = (self.time - old_time)/self.update_per
      
        display.clear_output(wait=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if 2*(len(self.losses)//2)-1 > 3:
            losses_smoothed = savgol_filter(self.losses, min(2*(len(self.losses)//2)-1, 51), 3)
        else:
            losses_smoothed = self.losses
            
        ts = np.arange(len(self.losses), dtype=np.int)+1
        ax.plot(ts, losses_smoothed, color=f"C0")
        ax.plot(ts, self.losses, color=f"C0", alpha=0.3)
        ax.set_xlabel(f"Epoch ({iter_time:.2g} s/epoch)")
        ax.set_ylabel("$-ELBO$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        plt.show()

def plot_1D(start, end, model, n_repeats=1, num_points=50, figsize=(10, 6), data=None, batch_size=32):
    ts = np.linspace(0, 1, num_points)
    start = np.array(start)
    end = np.array(end)
    r = end - start
    xs = start + ts[:, None] * r[None, :]
    xs = torch.from_numpy(xs)

    test_loader = DataLoader(TensorDataset(xs), batch_size=batch_size)

    means = []
    stds = []
    for x, in tqdm(test_loader):
        x = x.float().to(model.device)
        z = []
        for i in range(n_repeats):
            z.append(model(x))
        z = torch.stack(z, dim=0)
        means.append(z.mean(dim=0))
        if n_repeats > 1:
            stds.append(z.std(dim=0))

    means = torch.cat(means).detach().cpu().numpy().squeeze()
    if n_repeats > 1:
      stds  = torch.cat(stds).detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if n_repeats > 1:
        ax.fill_between(ts, means-stds, means+stds, 
                        alpha=0.2, label="$\sigma[f(x)]$")
    ax.plot(ts, means, label="$\mathbb{E}\;[f(x)]$")

    if data != None:
        x, y = data
        x, y = np.array(x), np.array(y)
        x -= start
        x = (x @ r)/np.sum(r**2)  
        x = x.squeeze()
        mask = np.logical_and(x >= 0, x <= 1)
        ax.scatter(x[mask], y[mask], c='r', s=10)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$f(x^{(0)}+t(x^{(1)}-x^{(0)})$")
    ax.set_xlim(0, 1)
    ax.legend()

    plt.show()

def plot_2D(xlim, ylim, model, n_repeats=1, num_points=50, figsize=(10, 6), data=None, batch_size=32):
    xs = np.linspace(*xlim, num_points)
    ys = np.linspace(*ylim, num_points)
    X, Y = np.meshgrid(xs, ys)
    X = X.ravel()
    Y = Y.ravel()
    Z = torch.from_numpy(np.stack((X, Y), axis=1))

    test_loader = DataLoader(TensorDataset(Z), batch_size=batch_size)

    means = []
    stds = []
    for x, in tqdm(test_loader):
        x = x.float().to(model.device)
        z = []
        for i in range(n_repeats):
            z.append(model(x))
        z = torch.stack(z, dim=0)
        means.append(z.mean(dim=0))
        if n_repeats > 1:
            stds.append(z.std(dim=0))

    means = torch.cat(means).detach().cpu().numpy()
    means = means.reshape(len(ys), len(xs))
    if n_repeats > 1:
      stds  = torch.cat(stds).detach().cpu().numpy()
      stds = stds.reshape(len(ys), len(xs))
    X = X.reshape(len(ys), len(xs))
    Y = Y.reshape(len(ys), len(xs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    cm = ax.contourf(X, Y, means, alpha=0.2)
    fig.colorbar(cm, ax=ax)
    ax.set_title("$\mathbb{E}\;[f(x)]$")

    if n_repeats > 1:
        ax = axes[1]
        cm = ax.contourf(X, Y, stds, alpha=0.2)
        fig.colorbar(cm, ax=ax)
        ax.set_title("$\sigma[f(x)]$")
    else:
        axes[1].axis("off")
        axes = axes[:1]

    for ax in axes:
        if data != None:
            x = np.array(data)
            ax.scatter(x[:, 0], x[:, 1], s=8, c='r')
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axis('scaled')

    plt.show()