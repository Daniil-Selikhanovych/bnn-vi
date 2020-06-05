import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from tqdm import tqdm
import random

def get_rotation(theta):
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R


class CircleDataset(Dataset):
    def __init__(self, n_samples, n_centers=9, sigma=0.02, include_zero=True, target_label=2., seed = None):
        super().__init__()
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        self.include_zero = include_zero
        self.nus = []
        if include_zero:
            self.nus.append(torch.zeros(2))
        self.sigma = sigma
        for i in range(n_centers-include_zero):
            R = get_rotation(i*360/(n_centers-include_zero))
            self.nus.append(torch.tensor([1, 0] @ R, dtype=torch.float))
        classes = torch.multinomial(torch.ones(n_centers), n_samples, 
                                    replacement=True)
        
        data = []
        target = []
        for i in range(n_centers):
            n_samples_class = torch.sum(classes == i)
            if n_samples_class == 0:
                continue
            dist = MultivariateNormal(self.nus[i], 
                                      torch.eye(2)*self.sigma**2)
            data.append(dist.sample([n_samples_class.item()]))
            enc = torch.full((n_samples_class, n_centers), -target_label)
            enc[:, i] = target_label
            target.append(enc + sigma * torch.randn(n_samples_class)[:, None])
        self.data = torch.cat(data).float()
        self.target = torch.cat(target).float()
        
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def __len__(self):
        return self.data.shape[0]

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

def model_1d(data):
    real_labels = torch.sin(12*data) + 0.66*torch.cos(25*data) + 3
    return real_labels

def noise_labels_model(real_labels, sigma_noise, seed = None):
    loc = 0.   # mean zero
    scale = 1.
    normal = torch.distributions.Normal(loc, scale) # create a normal distribution object
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    x = normal.rsample([real_labels.shape[0]]) 
    real_labels = real_labels + x*sigma_noise
    return real_labels


def get_sample_regression(n_samples, noise = 0.1, seed = 42):
    """
    Returns (x_train, y_train), (x_true, y_true)
    """
    gaussian_centers = torch.Tensor([[-1.0/(2**0.5)], [1.0/(2**0.5)]])
    data_num = n_samples
    data_sigma_noise = noise
    sigma = 0.01
    init_cov_matrix = torch.eye(1)
    cov_matrix_default = sigma*init_cov_matrix

    data_1d = gaussian_mixture_data_sampling(gaussian_centers, 
                                            cov_matrix_default, 
                                            data_num, 
                                            seed)
    real_labels = model_1d(data_1d[:, 0])
    noise_labels =  noise_labels_model(real_labels, 
                                    sigma_noise = data_sigma_noise, 
                                    seed = seed).reshape((real_labels.shape[0], 1))
    range_for_real_labels = torch.linspace(-1, 1, steps = 1000)
    real_labels_range = model_1d(range_for_real_labels)

    # data, range_for_real_labels, real_labels, noise_labels,
    return (data_1d[:, 0], noise_labels[:, 0]), (range_for_real_labels, real_labels_range) 