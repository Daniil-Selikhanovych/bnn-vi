import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample

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

    def forward(self, x, y):
        return self.seq(x)

class MultilayerBayesian(Multilayer):
    def __init__(self, in_features, out_features, hidden_features=50, n_layers=1, dropout=None,
                 device=torch.device('cpu'), target_std=0.1):
        super().__init__(in_features, out_features, hidden_features, n_layers, dropout, device)
        self.target_std = target_std
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
        
    def forward(self, x, y=None):
        y_pr = self.seq(x)
        if y != None:
            with pyro.plate("data", y.shape[0]):
                pyro.sample("obs", dist.Normal(y, self.target_std).to_event(1), obs=y_pr)
        return y_pr.detach()

    def get_mean_std(self, x, n_repeats=50):
        x = x.to(self.device)
        z = []
        for i in range(n_repeats):
            z.append(self.forward(x))
        z = torch.stack(z, dim=0)
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