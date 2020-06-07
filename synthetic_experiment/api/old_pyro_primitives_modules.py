import torch
from torch import nn
import torch.nn.functional as F

import pyro
from pyro.distributions import Normal, Delta
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.abstract_infer import EmpiricalMarginal, TracePredictive

import pandas as pd

get_marginal = lambda traces, sites: EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

class BNN_1HL(nn.Module):
    def __init__(self, num_feature, num_hidden): 
        super().__init__()
        self.fc1 = nn.Linear(num_feature, num_hidden)
        self.out = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x)
        return out
 
    def model(self, x_data, y_data):
        # weight and bias priors
        w1_prior = Normal(loc = torch.zeros_like(self.fc1.weight), 
                        scale = torch.ones_like(self.fc1.weight)).independent(2)
        b1_prior = Normal(loc = torch.zeros_like(self.fc1.bias),
                        scale = torch.ones_like(self.fc1.bias)).independent(1)

        wout_prior = Normal(loc=torch.zeros_like(self.out.weight), 
                            scale=torch.ones_like(self.out.weight)).independent(2)
        bout_prior = Normal(loc=torch.zeros_like(self.out.bias), 
                          scale=torch.ones_like(self.out.bias)).independent(1)

        priors = {'fc1.weight': w1_prior, 'fc1.bias': b1_prior, 
                  'out.weight': wout_prior,'out.bias': bout_prior}
        # lift module parameters from neural net
        lifted_module = pyro.random_module("module", self, priors)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(x_data)):
            #run forward on regression_model
            prediction = lifted_reg_model(x_data)
            prediction_mean = prediction[:, 0]
            softplus = torch.nn.Softplus()
            prediction_var = softplus(prediction[:, 1])
            prediction_std = torch.pow(prediction_var, 0.5)
            # condition on the observed data
            pyro.sample("obs", Normal(prediction_mean, prediction_std), obs = y_data)
            return prediction_mean

    def guide(self, x_data, y_data):
        softplus = torch.nn.Softplus()
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(self.fc1.weight)
        fc1w_sigma = torch.randn_like(self.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(self.fc1.bias)
        fc1b_sigma = torch.randn_like(self.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        # Output layer weight distribution priors
        outw_mu = torch.randn_like(self.out.weight)
        outw_sigma = torch.randn_like(self.out.weight)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc = outw_mu_param, scale = outw_sigma_param).independent(1)
        # Output layer bias distribution priors
        outb_mu = torch.randn_like(self.out.bias)
        outb_sigma = torch.randn_like(self.out.bias)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
        
        lifted_module = pyro.random_module("module", self, priors)
        
        return lifted_module()

    def train(self, x_train, y_train, num_epoch = 80000, 
              lr = 1e-2, num_samples = 128, every_epoch_to_print = 100):
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, 
                  loss = Trace_ELBO(), num_samples = num_samples)
        pyro.clear_param_store()
        loss_arr = []
        for j in range(num_epoch):
            # calculate the loss and take a gradient step
            loss = svi.step(x_train, y_train)
            loss_arr.append(loss)
            if j % every_epoch_to_print == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))
 
        return loss_arr, svi

    def wrapped_model(self, x_data, y_data):
        pyro.sample("prediction", Delta(self.model(x_data, y_data)))

    def sampling_prediction(self, svi, x_train, y_train, x_test, 
                            num_samples = 1000):
        posterior = svi.run(x_train, y_train)
        trace_pred = TracePredictive(self.wrapped_model, posterior,  
                                      num_samples = num_samples)
        post_pred = trace_pred.run(x_test, None)
        sites= ['prediction', 'obs']
        marginal = get_marginal(post_pred, sites)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            site_stats[site_name] = marginal_site.apply(pd.Series.describe, 
                                                        axis=1)[["mean", "std"]]

        mu = site_stats["prediction"]
        y_o = site_stats["obs"]

        return mu["mean"], mu["std"], y_o["mean"], y_o["std"] 
