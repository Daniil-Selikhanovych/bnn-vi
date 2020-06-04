# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:30:35 2020

@author: a
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Function
from matplotlib import pyplot as plt
from itertools import product

EPS = 1e-4
Sigma = {1:4,2:3,3:2.25,4:2,5:2,6:1.9,7:1.75,8:1.75,9:1.7,10:1.65}
p = 0.05

class MCDO(nn.Module):
  def __init__(self,in_dim,out_dim,n_layers = 1,hid_dim=50,p=0.05):
    super().__init__()
    self.n_layers = n_layers

    self.linear_in = nn.Linear(in_dim,hid_dim)
    nn.init.normal_(self.linear_in.weight,std = 1/(4*hid_dim)**0.5)
    nn.init.zeros_(self.linear_in.bias)


    self.dropout_in = nn.Dropout(p)

    if n_layers>1:
      models = list(range(3*(n_layers-1)))
      for i in range(0,len(models),3):
        models[i]=nn.Linear(hid_dim,hid_dim)
        nn.init.normal_(models[i].weight,std = 1/(4*hid_dim)**0.5)
        nn.init.zeros_(models[i].bias)

      for i in range(1,len(models),3):
        models[i]=nn.ReLU()
      for i in range(2,len(models),3):
        models[i]=nn.Dropout(p)
      
      self.hid_layers = nn.Sequential(*models)

    self.linear_out = nn.Linear(hid_dim,out_dim)
    nn.init.normal_(self.linear_out.weight,std = 1/(4*out_dim)**0.5)
    nn.init.zeros_(self.linear_out.bias)

  def forward(self,x):
    x = torch.relu(self.linear_in(x))

    x = self.dropout_in(x)

    if self.n_layers>1: x = self.hid_layers(x)
    
    x = self.linear_out(x)
    return x




def MCDO_loss(model,x,y, depth=1,n_samples = 32, tau = 1,H = 2):
  res = 0
  N = y.shape[0]
  l = np.sqrt(H)/Sigma[depth]
  lambd = p*l**2/N/tau
  for module in model.modules():
    if type(module).__name__=='Linear':
      W = module.weight
      b = module.bias
      res+=torch.pow(torch.norm(W,p=2),2)+torch.pow(torch.norm(b,p=2),2)
  res *= lambd
  pred = torch.cat([model(x) for i in range(n_samples)],dim=1).mean(dim=1)
  res += torch.pow(pred-y,2).mean()
  return res





class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, mu_W, mu_b, rho_W, rho_b):

        z_W = torch.normal(torch.zeros_like(mu_W), torch.ones_like(rho_W))
        z_b = torch.normal(torch.zeros_like(mu_b), torch.ones_like(rho_b))

        W = mu_W+z_W*torch.log(1+EPS+torch.exp(rho_W))
        ctx.save_for_backward(input,W, mu_W, mu_b, rho_W, rho_b,z_W,z_b)

        b = mu_b+z_b*torch.log(1+EPS+torch.exp(rho_b))

        output = torch.mm(input,W)+b

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input,W, mu_W, mu_b, rho_W, rho_b,z_W,z_b = ctx.saved_tensors

        grad_input = grad_mu_W = grad_mu_b = grad_rho_W = grad_rho_b = None

        grad_input = grad_output.mm(W.t())

        grad_mu_W = input.t().mm(grad_output)

        grad_mu_b = grad_output.sum(0)

        ex_W = torch.exp(rho_W)
        grad_rho_W = grad_mu_W*z_W*ex_W*torch.pow(1+EPS+ex_W,-1)

        ex_b = torch.exp(rho_b)
        grad_rho_b = grad_mu_b*z_b*ex_b*torch.pow(1+EPS+ex_b,-1)

        return grad_input, grad_mu_W, grad_mu_b, grad_rho_W, grad_rho_b
    
    
    
class Random_Linear(nn.Module):
  def __init__(self,in_dim,out_dim):

    super().__init__()

    temp_W = torch.ones(in_dim,out_dim)

    self.mu_W = torch.nn.Parameter( torch.normal(temp_W*0, temp_W/np.sqrt(4*out_dim)) )

    self.mu_b = torch.nn.Parameter(torch.zeros(out_dim))

    rho_W = np.log(np.exp(10**(-2.5))-1)

    self.rho_W = torch.nn.Parameter(torch.ones(in_dim,out_dim)*rho_W)

    rho_b = np.log(np.exp(10**(-2.5))-1)

    self.rho_b = torch.nn.Parameter(torch.ones(out_dim)*rho_b)

  def forward(self,x):
    # if self.training:
    return LinearFunction.apply(x, self.mu_W, self.mu_b, self.rho_W, self.rho_b)
    # output = torch.mm(x,self.mu_W)+self.mu_b
    # return output
    

  
    
    
class Bayesian_ReLU(nn.Module):
  def __init__(self,in_dim,out_dim,n_layers = 1,hid_dim=50):
    super().__init__()
    self.n_layers = n_layers
    self.linear_in = Random_Linear(in_dim,hid_dim)

    if n_layers>1:
      models = list(range(2*(n_layers-1)))
      for i in range(0,len(models),2):
        models[i]=Random_Linear(hid_dim,hid_dim)
      for i in range(1,len(models),2):
        models[i]=nn.ReLU()
      self.hid_layers = nn.Sequential(*models)

    self.linear_out = Random_Linear(hid_dim,out_dim)

  def forward(self,x):

    x = torch.relu(self.linear_in(x))
    if self.n_layers>1: x = self.hid_layers(x)
    x = self.linear_out(x)
    return x



def KL(model,depth, hid_dim = 50):
  res = 0
  sigma_W_init = Sigma[depth]/hid_dim**0.5
  sigma_b_init = 1

  for module in model.modules():
    if type(module).__name__=='Random_Linear':
      mu_W = module.mu_W
      mu_b = module.mu_b

      sigma_W = torch.log(1+EPS+torch.exp(module.rho_W))
      sigma_b = torch.log(1+EPS+torch.exp(module.rho_b))

      res+=1/2*(torch.pow(sigma_W/sigma_W_init,2)+torch.pow(mu_W/sigma_W_init,2)+2*torch.log(torch.pow(sigma_W,-1)*sigma_W_init)).sum()
      res+=1/2*(torch.pow(sigma_b/sigma_b_init,2)+torch.pow(mu_b/sigma_b_init,2)+2*torch.log(torch.pow(sigma_b,-1)*sigma_b_init)).sum()
  return res



def mean_reduction(preds,y):
  preds[:,[0]]-=y
  return preds



def elbo_loss(model,x,y,depth,TRAIN_LENGTH, n_samples = 32):
  bs = x.shape[0]
  res = torch.cat([mean_reduction(model(x),y) for i in range(n_samples)])
  mu = res[:,0]
  mask = (res[:,1]<=60).int()
  std = torch.log( 1+EPS+torch.exp(res[:,1]*mask) )+res[:,1]*(1-mask)
  dens = -torch.pow(mu/std,2)/2 - torch.log(std)
  res =  KL(model,depth)*bs/TRAIN_LENGTH-dens.sum()/n_samples
  return res



def loss_vi(model, x,mu_target,var_target,n_samples = 32):
  bs = x.shape[0]
  preds = torch.cat([model(x) for i in range(n_samples)])
  mu = preds[:,0]
  mask = (preds[:,1]<=60).int()
  var = torch.pow(torch.log(1+EPS+torch.exp(preds[:,1]*mask))+preds[:,1]*(1-mask),2)

  mu=mu.view(n_samples,bs).T.mean(dim=1)
  var=var.view(n_samples,bs).T.mean(dim=1)

  mu_target = mu_target.view(mu.shape)
  var_target = var_target.view(var.shape)

  mu_diff = mu-mu_target
  var_diff = var-var_target

  res = mu_diff@mu_diff+var_diff@var_diff

  return res



def loss_mcdo(model, x,mu_target,var_target,n_samples = 32):
  bs = x.shape[0]
  preds = torch.cat([model(x) for i in range(n_samples)],dim=1)

  mu = preds.mean(dim=1)
  var = preds.var(dim=1)

  mu_target = mu_target.view(mu.shape)
  var_target = var_target.view(var.shape)

  mu_diff = mu-mu_target
  var_diff = var-var_target

  res = mu_diff@mu_diff +var_diff@var_diff

  return res