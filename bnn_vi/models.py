from functools import partial

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
        self.old_guide = None
        
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

    def train(self, data_loader, n_epochs, lr=1e-3, log_per=5, show_smooth=True):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []
        fig = None
        pp = ProgressPlotter(losses, "MSE", log_per, show_smooth)
        pp.start()
        for epoch in range(n_epochs):
            total_loss = 0.
            for x, y in data_loader:
                x, y = x.float().to(self.device), y.float().to(self.device)
                loss = criterion(self.forward(x), y.reshape(y.shape[0], -1))
                total_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()

            total_loss /= len(data_loader)
            losses.append(total_loss)
            fig = pp.update(epoch)
        
        return pp.fig, losses

    def get_mean_std(self, x):
        out = self.forward(x.reshape(x.shape[0], -1))
        return out, torch.zeros_like(out)


def init_loc(x):
    if 'weight' in x['name']:
        n_out = x['fn'].base_dist.loc.shape[0]
        means =  torch.randn_like(x['fn'].base_dist.loc)/(4*n_out)**0.5
    elif 'bias' in x['name']:
        means = torch.zeros_like(x['fn'].base_dist.loc)
    return means

class MultilayerBayesian(Multilayer):
    def __init__(self, in_features, out_features, hidden_features=50, n_layers=1, dropout=None,
                 device=torch.device('cpu'), target_std=0.1):
        super().__init__(in_features, out_features, hidden_features, n_layers, dropout, device)
        pyro.nn.module.to_pyro_module_(self)
        self.target_std = target_std
         # See the paper https://arxiv.org/pdf/1909.00719.pdf
        # stds = [4., 3., 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]
        stds = [10., 3., 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]
        self.to(device)
        for i in range(len(self.seq)):
            if 'linear' in type(self.seq[i]).__name__.lower():
                out_size, in_size = self.seq[i].weight.shape 
                # We can't specify the device explicitly, thus using this hack
                self.seq[i].bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.,
                                                          validate_args=False).expand([out_size]).to_event(1))
                self.seq[i].weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 
                                                            stds[n_layers-1]/in_size**0.5, validate_args=False).expand([out_size,
                                                                                          in_size]).to_event(2))

    def load_pretrained(self, model):
        weights = dict(model.named_parameters())
        self.guide = AutoDiagonalNormal(self, init_loc_fn=lambda x: weights[x['name']], init_scale=1e-5)

        
    def forward(self, x, y=None):
        y_pr = self.seq(x)
        if y != None:
            with pyro.plate("data", y.shape[0]):
                pyro.sample("obs", dist.Normal(y, self.target_std).to_event(1), obs=y_pr)
        return y_pr.detach()

    def train(self, data_loader, n_epochs, num_particles=1, lr=1e-3, log_per=5, show_smooth=True, save_per=10):
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(self, init_loc_fn=init_loc, init_scale=1e-5)
        self.guide = guide
        svi = SVI(self, guide, Adam({"lr": lr}), TraceMeanField_ELBO(num_particles=num_particles))
        losses = []
        fig = None
        pyro.clear_param_store()
        pp = ProgressPlotter(losses, "$-ELBO$", log_per, show_smooth)
        pp.start()
        for epoch in range(n_epochs):
            total_loss = 0.
            for x, y in data_loader:
                x, y = x.float().to(self.device), y.float().to(self.device)
                loss = svi.step(x, y) / y.numel()
                total_loss += loss
            total_loss /= len(data_loader)
            losses.append(total_loss)
            fig = pp.update(epoch)
            if epoch % save_per == 1:
                self.save('MultilayerBayesian.pth')
        return pp.fig, losses

    def get_mean_std(self, x, n_repeats=50):
        predictive = Predictive(self, guide=self.guide, num_samples=n_repeats,
                        return_sites=("_RETURN", ))
        x = x.to(self.device)
        z = predictive(x)['_RETURN']
        return z.mean(dim=0), z.std(dim=0)

    def save(self, filename):
        state = {'state_dict': self.state_dict(),
                 'params': pyro.get_param_store().get_state()
                }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state_dict'])
        self.guide = state['guide']
        pyro.get_param_store().set_state(state['params']) 

    def condition(self, data):
        self.guide = pyro.condition(AutoDiagonalNormal(self, init_loc_fn=init_loc, init_scale=1e-5), data)

    def uncondition(self):
        self.guide = AutoDiagonalNormal(self, init_loc_fn=init_loc, init_scale=1e-5)

def guide(x, y=None, model=None):
    for i, module in enumerate(model.seq):
        if isinstance(module, nn.Linear):
            n_out = module.weight.shape[0]
            weight_mean = pyro.param(f"linear{i}_weights_mean",
                                    torch.randn_like(module.weight)/(4*n_out)**0.5)
            weight_std  = pyro.param(f"linear{i}_weights_std",
                                    torch.ones_like(module.weight)*1e-5)
            bias_mean = pyro.param(f"linear{i}_bias_mean",
                                torch.zeros_like(module.bias))
            bias_std  = pyro.param(f"linear{i}_bias_std",
                                torch.ones_like(module.bias)*1e-5)
            new_mean = x @ weight_mean.t() + bias_mean
            new_std2 = x**2 @ weight_std.t()**2 + bias_std**2
            x = pyro.sample(f"linear{i}_out", 
                            dist.Normal(new_mean, new_std2**0.5))
        else:
            x = module(x)

    return x



class MultilayerBayesianTrick(Multilayer):
    def __init__(self, in_features, out_features, hidden_features=50, n_layers=1, dropout=None,
                 device=torch.device('cpu'), target_std=0.1):
        super().__init__(in_features, out_features, hidden_features, n_layers, dropout, device)
        self.target_std = target_std
        self.guide = None
        # pyro.nn.module.to_pyro_module_(self)
         # See the paper https://arxiv.org/pdf/1909.00719.pdf
        stds = [1000., 3., 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]
        self.to(device)
        for i in range(len(self.seq)):
            if isinstance(self.seq[i], nn.Linear):
                out_size, in_size = self.seq[i].weight.shape 
                self.seq[i].bias_mean = torch.zeros_like(self.seq[i].bias)
                self.seq[i].bias_std  = torch.ones_like(self.seq[i].bias)

                self.seq[i].weight_mean = torch.zeros_like(self.seq[i].weight)
                self.seq[i].weight_std  = torch.ones_like(self.seq[i].weight)*stds[n_layers-1]/in_size**0.5
        
    def forward(self, x, y=None):
        for i, module in enumerate(self.seq):
            if isinstance(module, nn.Linear):
                new_mean = x @ module.weight_mean.t() + module.bias_mean
                new_std2 = x**2 @ module.weight_std.t()**2 + module.bias_std**2
                x = pyro.sample(f"linear{i}_out", 
                                dist.Normal(new_mean, new_std2**0.5))
            else:
                x = module(x)

        if y != None:
            with pyro.plate("data", y.shape[0]):
                pyro.sample("obs", dist.Normal(y, self.target_std).to_event(1), obs=x)

        return x

    def load_pretrained(self, model):
        pyro.clear_param_store()
        params = dict(model.named_parameters())
        for i, module in enumerate(self.seq):
            if isinstance(module, nn.Linear):
                pyro.param(f"linear{i}_weights_mean",
                            params[f"seq.{i}.weight"])
                pyro.param(f"linear{i}_weights_std",
                            torch.ones_like(module.weight)*1e-5)
                pyro.param(f"linear{i}_bias_mean",
                            params[f"seq.{i}.bias"])
                pyro.param(f"linear{i}_bias_std",
                            torch.ones_like(module.bias)*1e-5)

    def train(self, data_loader, n_epochs, num_particles=1, lr=1e-3, log_per=5, show_smooth=True, save_per=10):
        pyro.clear_param_store()
        self.guide = partial(guide, model=self)
        svi = SVI(self, self.guide, Adam({"lr": lr}), TraceMeanField_ELBO(num_particles=num_particles))
        losses = []
        fig = None
        pp = ProgressPlotter(losses, "$-ELBO$", log_per, show_smooth)
        pp.start()
        for epoch in range(n_epochs):
            total_loss = 0.
            for x, y in data_loader:
                x, y = x.float().to(self.device), y.float().to(self.device)
                loss = svi.step(x, y) / y.numel()
                total_loss += loss
            total_loss /= len(data_loader)
            losses.append(total_loss)
            fig = pp.update(epoch)
            if epoch % save_per == 1:
                self.save('MultilayerBayesianTrick.pth')
        return pp.fig, losses

    def condition(self, data):
        self.guide = pyro.condition(partial(guide, model=self), data)

    def uncondition(self):
        self.guide = partial(guide, model=self)

    def get_mean_std(self, x, n_repeats=50):
        predictive = Predictive(self, guide=self.guide, num_samples=n_repeats,
                                return_sites=("_RETURN", ))
        x = x.to(self.device)
        z = predictive(x)['_RETURN']
        return z.mean(dim=0), z.std(dim=0)

    def save(self, filename):
        state = {'state_dict': self.state_dict(),
                 'params': pyro.get_param_store().get_state()
                }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state_dict'])
        self.guide = state['guide']
        pyro.get_param_store().set_state(state['params'])        

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
            data = {name: values[i] for name, values in parameters_samples.items()}
            model.condition(data)
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
        for i in range(len(self.models)):
            mean, _ = self.models[i].get_mean_std(x, n_repeats=1)
            ys.append(mean)
        ys = torch.stack(ys)
 
        return ys.mean(dim=0), ys.std(dim=0)

    def forward(self, x):
        mean, std = self.get_mean_std(x)

        return mean