import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import random

def gaussian_sampler_2d(gaussian_center, cov_matrix):
    mu_distr = MultivariateNormal(gaussian_center, cov_matrix)
    return mu_distr

def gaussian_data_sampling(gaussian_center, cov_matrix, data_num, seed = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    sampler = gaussian_sampler_2d(gaussian_center, cov_matrix)
    data = sampler.sample(sample_shape=torch.Size([data_num]))

    return data
    
def gaussian_mixture_data_sampling(centers, cov_matrix, data_num, seed = None, device = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    index_to_choice = np.random.randint(centers.shape[0], size = data_num)
    data_clusters = gaussian_data_sampling(centers[index_to_choice[0]], cov_matrix, 1)
    for i in range(1, data_num):
        cur_data = gaussian_data_sampling(centers[index_to_choice[i]], cov_matrix, 1)
        data_clusters = torch.cat((data_clusters, cur_data), 0)

    return data_clusters
