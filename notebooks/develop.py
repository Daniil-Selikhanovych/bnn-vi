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

if '..' not in sys.path:
    sys.path.append('..')

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

    N_WORKERS = 0 if os.name != 'nt' else 0# no threads for windows :c
    if torch.cuda.is_available():
        print("Using CUDA!")
        DEVICE = torch.device('cuda')
        IS_CUDA = True
    else:
        print("Using CPU!")
        DEVICE = torch.device('cpu')
        IS_CUDA = False

    DEVICE = torch.device('cpu')
    IS_CUDA = False
    N_SAMPLES = 100

    circles = CircleDataset(N_SAMPLES, 2, sigma=0.2, ysigma=0.1, include_zero=False)
    circles = TensorDataset(circles.data @ get_rotation(-45),
                            circles.target[:, 0])

    (x_reg, y_reg), (x_true, y_true) = get_sample_regression(N_SAMPLES)
    regression = TensorDataset(x_reg, y_reg)


    # data = circles
    data = regression

    pyro.set_rng_seed(1)

    params = {
        'in_features': data.tensors[0].reshape(N_SAMPLES, -1).shape[1],
        'out_features': 1,
        'hidden_features': 50,
        'n_layers': 1,
        'dropout': None,
        'device': DEVICE
    }
    target_std = 1e-1
    model_b = MultilayerBayesian(**params, target_std=target_std)


    DO_LOAD = False

    samples = None
    if DO_LOAD:
        samples = torch.load('samples_reg.pth', map_location='cpu')
    else:
        pyro.clear_param_store()
        nuts_kernel = NUTS(model_b)

        mcmc = MCMC(nuts_kernel, num_samples=300, num_chains=5)
        mcmc.run(data.tensors[0].reshape(N_SAMPLES, -1).float().to(DEVICE), data.tensors[1].reshape(N_SAMPLES, -1).float().to(DEVICE))
        samples = mcmc.get_samples()
        torch.save(samples, 'samples_reg.pth')