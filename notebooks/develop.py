#!/usr/bin/env python
# coding: utf-8

# [Git](https://github.com/Daniil-Selikhanovych/bnn-vi)
# 
# [On the Expressiveness of Approximate Inference in Bayesian Neural Networks](https://arxiv.org/pdf/1909.00719.pdf)
# 
# [Sufficient Conditions for Idealised Models to Have No
# Adversarial Examples: a Theoretical and Empirical
# Study with Bayesian Neural Networks](https://arxiv.org/pdf/1806.00667.pdf)
# 
# [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf)
# 
# [Neural Networks as Gaussian Process](https://arxiv.org/pdf/1711.00165.pdf)
# 
# [VAE in Pyro](https://pyro.ai/examples/svi_part_i.html)
# 
# [Neural Networks in Pyro](http://docs.pyro.ai/en/stable/nn.html)
# 
# [Bayessian Regression in Pyro](https://pyro.ai/examples/bayesian_regression_ii.html?highlight=sample)
# 
# [Intro to HMC](https://arxiv.org/pdf/1206.1901.pdf)
# 
# [Stochastic HMC](https://arxiv.org/pdf/1402.4102.pdf)
# 
# ![details1](../img/description1.png)
# ![details2](../img/description2.png)

# In[1]:


import os, sys
import random
import copy
from itertools import islice
from functools import partial
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import torch
from torch.utils.data import DataLoader, TensorDataset

import pyro
from pyro.infer import (SVI, Trace_ELBO, TraceMeanField_ELBO,
                        MCMC, NUTS
                       )
from pyro.infer.autoguide import (AutoNormal, AutoDiagonalNormal, 
                                  AutoMultivariateNormal
                                 )
from pyro.optim import Adam


# In[2]:


rcParams.update({'font.size': 12})


# In[3]:

if '..' not in sys.path:
    sys.path.append('..')
# In[4]:


from bnn_vi.data_sampling import (CircleDataset,
                                  get_rotation,
                                  get_sample_regression
)
from bnn_vi.models import (Multilayer,
                           MultilayerBayesian,
                           ModelGenerator,
)
from bnn_vi.plotting import (ProgressPlotter,
                          plot_1D,
                          plot_2D
                         )

if __name__ == "__main__":
    torch.manual_seed(42)


    # In[5]:

    # In[6]:


    DEVICE = torch.device('cpu')
    IS_CUDA = False


    # In[7]:


    BATCH_SIZE = 100
    N_WORKERS = 0 if os.name != 'nt' else 0# no threads for windows :c


    # In[8]:


    circles = CircleDataset(BATCH_SIZE, 2, sigma=0.12, target_label=2., include_zero=False)
    circles.data @= get_rotation(-45)


    # In[9]:
    (x_reg, y_reg), (x_true, y_true) = get_sample_regression(BATCH_SIZE)

    gauss_loader = DataLoader(circles, batch_size=BATCH_SIZE, pin_memory=IS_CUDA,
                            shuffle=True, num_workers=N_WORKERS)


    # In[10]:


    pyro.set_rng_seed(1)

    N_EPOCHS = 1000
    log_per = 5

    params = {
        'in_features': 2,
        'out_features': 1,
        'hidden_features': 50,
        'n_layers': 1,
        'dropout': None,
        'device': DEVICE
    }

    model = Multilayer(**params)
    model_b = MultilayerBayesian(**params)


    # In[11]:


    DO_LOAD = False


    # In[12]:


    samples = None
    if DO_LOAD:
        samples = torch.load('samples.pth', map_location='cpu')
    else:
        pyro.clear_param_store()
        nuts_kernel = NUTS(model_b)

        mcmc = MCMC(nuts_kernel, num_samples=400, warmup_steps=200, num_chains=5) 
        # mcmc.run(x_reg[:, None].float().to(DEVICE), y_reg[:, None].float().to(DEVICE))
        mcmc.run(circles.data.float().to(DEVICE), circles.target[:, 0].float().to(DEVICE))
        samples = mcmc.get_samples()
        torch.save(samples, 'samples.pth')