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
    def __init__(self, losses, ylabel="Loss", update_per=5, show_smooth=True):
        self.losses = losses
        self.ylabel = ylabel
        self.update_per = update_per
        self.fig = None
        self.show_smooth = show_smooth
        
    def start(self):
        self.time = time.time()
        
    def update(self, num_iter):
        if num_iter % self.update_per != 0:
            return
        old_time = self.time
        self.time = time.time()
        iter_time = (self.time - old_time)/self.update_per
      
        display.clear_output(wait=True)
        self.fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if 2*(len(self.losses)//2)-1 > 3:
            losses_smoothed = savgol_filter(self.losses, min(2*(len(self.losses)//2)-1, 51), 3)
        else:
            losses_smoothed = self.losses
            
        ts = np.arange(len(self.losses), dtype=np.int)+1
        if self.show_smooth:
            ax.plot(ts, self.losses, color=f"C0", alpha=0.3, label="Loss")
            ax.plot(ts, losses_smoothed, color=f"C0", label="Smoothed loss")
            ax.legend()
        else:
            ax.plot(ts, self.losses, color=f"C0")
        ax.set_xlabel(f"Epoch ({iter_time:.2g} s/epoch)")
        ax.set_ylabel(self.ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        plt.show()

def plot_1D(start, end, model, num_points=50, figsize=(10, 6), data_scatter=None, data_plot=None,
            batch_size=32):
    """
    Mode is 'class' ~ classification or 'reg' ~ regression
    """
    ts = np.linspace(0, 1, num_points)
    start = np.array(start)
    end = np.array(end)
    r = end - start
    xs = start + ts[:, None] * r[None, :]
    xs = torch.from_numpy(xs)

    test_loader = DataLoader(TensorDataset(xs), batch_size=batch_size)

    means = []
    stds = []
    for i, (x, ) in tqdm(enumerate(test_loader)):
        mean, std = model(x.float())
        means.append(mean)
        stds.append(std)

    means = torch.cat(means).detach().cpu().numpy().squeeze()
    stds  = torch.cat(stds).detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.fill_between(ts, means-stds, means+stds, 
                    alpha=0.2, label="$\sigma[f(x)]$")
    ax.plot(ts, means, label="$\mathbb{E}\;[f(x)]$")

    data = {'scatter': data_scatter,
            'plot': data_plot}
    for key in data:
        if data[key] == None:
            continue
        x, y = data[key]
        x, y = np.array(x), np.array(y)
        x -= start
        norm = None
        if r.size > 1:
            x = (x @ r)/np.sum(r**2)
        else:
            x = (x * r)/r**2
        
        mask = np.logical_and(x >= 0, x <= 1)
        data[key] = (x[mask], y[mask])
    if data['scatter'] != None:
        x, y = data['scatter']
        ax.scatter(x, y, c='r', s=10, alpha = 0.5, label='Source data')
    if data['plot'] != None:
        x, y = data['plot']
        ax.plot(x, y, c='r', alpha = 0.4)
        
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f(x^{(0)}+t(x^{(1)}-x^{(0)})$")
    ax.set_xlim(0, 1)
    ax.legend()

    plt.show()
    return fig

def plot_2D(xlim, ylim, model, num_points=50, figsize=(10, 6), data=None, batch_size=32):
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
        mean, std = model(x.float())
        means.append(mean)
        stds.append(std)

    means = torch.cat(means).detach().cpu().numpy()
    means = means.reshape(len(ys), len(xs))
    stds  = torch.cat(stds).detach().cpu().numpy()
    stds = stds.reshape(len(ys), len(xs))
    X = X.reshape(len(ys), len(xs))
    Y = Y.reshape(len(ys), len(xs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    cm = ax.contourf(X, Y, means, alpha=0.2)
    fig.colorbar(cm, ax=ax)
    ax.set_title("$\mathbb{E}\;[f(x)]$")

    ax = axes[1]
    cm = ax.contourf(X, Y, stds, alpha=0.2)
    fig.colorbar(cm, ax=ax)
    ax.set_title("$\sigma[f(x)]$")

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
    return fig