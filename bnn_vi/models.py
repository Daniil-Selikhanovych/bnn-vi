import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer import (SVI, Trace_ELBO, TraceMeanField_ELBO,
                        MCMC, NUTS, Predictive
                       )
from pyro.infer.autoguide import (AutoNormal, AutoDiagonalNormal, 
                                  AutoMultivariateNormal
                                 )
from pyro.optim import Adam

from .plotting import ProgressPlotter

class Multilayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=50, n_layers=1, dropout=None,
                 device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        
        pipe = [nn.Linear(in_features, hidden_features),
                nn.ReLU()]
        for i in range(n_layers-1):
            pipe += [nn.Linear(hidden_features, hidden_features),
                     nn.ReLU()]
            if dropout != None:
                pipe += [nn.Dropout(p=dropout)]
            
        pipe += [nn.Linear(hidden_features, out_features)]
        if dropout != None:
            pipe += [nn.Dropout(p=dropout)]
        self.seq = nn.Sequential(*pipe)
        self.to(device)

    def forward(self, x):
        return self.seq(x)

def init_loc(x):
    n_out = x['fn'].base_dist.loc.shape[0]
    return torch.randn_like(x['fn'].base_dist.loc)/(4*n_out)**0.5

class MultilayerBayesian(Multilayer):
    def __init__(self, in_features, out_features, hidden_features=50, n_layers=1, dropout=None,
                 device=torch.device('cpu'), target_std=0.1):
        super().__init__(in_features, out_features, hidden_features, n_layers, dropout, device)
        self.target_std = target_std
        self.guide = None
        pyro.nn.module.to_pyro_module_(self)
         # See the paper https://arxiv.org/pdf/1909.00719.pdf
        stds = [4., 3., 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]
        stds = stds[::-1][:n_layers+1][::-1]
        k = -1
        self.to(device)
        for i in range(len(self.seq)):
            if 'linear' in type(self.seq[i]).__name__.lower():
                k += 1
                out_size, in_size = self.seq[i].weight.shape 
                # We can't specify the device explicitly, thus using this hack
                self.seq[i].bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.,
                                                          validate_args=False).expand([out_size]).to_event(1))
                self.seq[i].weight = PyroSample(dist.Normal(torch.tensor(0., device=device),
                                                            stds[k]/hidden_features**0.5, validate_args=False).expand([out_size,
                                                                                          in_size]).to_event(2))

        guide = AutoDiagonalNormal(self, init_loc_fn=init_loc, init_scale=1e-5)
        self.guide = guide
        
    def forward(self, x, y=None):
        y_pr = self.seq(x)
        if y != None:
            with pyro.plate("data", y.shape[0]):
                pyro.sample("obs", dist.Normal(y, self.target_std).to_event(1), obs=y_pr)
        return y_pr.detach()

    def train(self, data_loader, n_epochs, num_particles=1, lr=1e-3, log_per=5):
        svi = SVI(self, self.guide, Adam({"lr": lr}), TraceMeanField_ELBO(num_particles=num_particles))
        losses = []
        pyro.clear_param_store()
        pp = ProgressPlotter(losses, log_per)
        pp.start()
        for epoch in range(n_epochs):
            total_loss = 0.
            for i, batch in enumerate(data_loader):
                x, y = batch
                x, y = x.float().to(self.device)[:, None], y.float().to(self.device)
                loss = svi.step(x, y) / y.numel()
                total_loss += loss
            total_loss /= len(data_loader)
            losses.append(total_loss)
            pp.update(epoch)

        return losses

    def get_mean_std(self, x, n_repeats=50):
        predictive = Predictive(self, guide=self.guide, num_samples=n_repeats,
                        return_sites=("_RETURN", ))
        x = x.to(self.device)
        z = predictive(x)['_RETURN']
        return z.mean(dim=0), z.std(dim=0)

class ModelGenerator:
    def __init__(self, creator, *args, **kwargs):
        """
        Creates models from the dictionary with its parameters.
        Such dictionaries are provided by MCMC samples, for example.
        """
        self.creator = creator
        self.args = args
        self.kwargs = kwargs

    def __call__(self, parameters_samples):
        """
        Generates list of model given the dictionary of parameters samples.
        """
        n_models = len(next(iter(parameters_samples.values())))
        models = []
        for i in range(n_models):
            model = self.creator(*self.args, **self.kwargs)
            for name, p in model.named_parameters():
                p.data[...] = parameters_samples[name][i]
            models.append(model)
        
        return models

class EstimatorPool(nn.Module):
    def __init__(self, models, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList(models)

    def get_mean_std(self, x):
        x = x.to(self.device)
        ys = []
        for model in self.models:
            ys.append(model(x))
        ys = torch.stack(ys)

        return ys.mean(dim=0), ys.std(dim=0)

    def forward(self, x):
        mean, std = self.get_mean_std(x)

        return mean