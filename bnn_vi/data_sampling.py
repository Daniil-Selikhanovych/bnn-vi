import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from tqdm import tqdm

def get_rotation(theta):
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R


class CircleDataset(Dataset):
    def __init__(self, n_samples, n_centers=9, sigma=0.02, include_zero=True, target_label=2.):
        super().__init__()
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
    
def plot_2d_data(data, figsize = (12, 6)):
    fig = plt.figure(figsize = figsize)

    plt.xlabel(r'$x_1$') 
    plt.ylabel(r'$x_2$') 
    plt.title('Synthetic data') 

    plt.scatter(data[:, 0], data[:, 1], label = 'data clusters', marker='+', color = 'r')

    plt.legend()
    plt.grid(True) 
    plt.show()
    
def plot_1d_label_data(data, range_for_real_labels, 
                       real_labels, noise_labels, 
                       model_name, 
                       range_for_prediction = None,
                       mean_prediction = None,
                       var_prediction = None,
                       model_prediction_name = None,
                       coef = 2.0,
                       figsize = (12, 6),
                       loc = "best"):
    fig = plt.figure(figsize = figsize)

    plt.xlabel(r'$x$') 
    plt.ylabel(r'$y$') 
    plt.title(fr'Synthetic data, model: {model_name}') 

    plt.scatter(data, noise_labels, label = 'noise data', marker='+', color = 'r')
    plt.plot(range_for_real_labels, real_labels, label = 'real labels')
    
    if ((range_for_prediction is not None) and
        (mean_prediction is not None) and
        (var_prediction is not None) and
        (model_prediction_name is not None)):
        plt.plot(range_for_prediction, mean_prediction, c = 'orange', label = f'mean {model_prediction_name} prediction')
        plt.fill_between(range_for_prediction, mean_prediction - coef * np.sqrt(var_prediction),
                         mean_prediction + coef * np.sqrt(var_prediction), color="C0",
                         alpha=0.2, label = fr'confidence interval for {model_prediction_name}: mean $\pm \; 2 \cdot std$')
    plt.legend(loc = loc)
    plt.grid(True)
    plt.show() 

def plot_variance_and_mean(grid, target_variance = None, target_mean = None, 
                           prediction_variance = None, prediction_mean = None,
                           figsize = (12, 6), comparison_method = 'BNN'):
    fig = plt.figure(figsize = figsize)
    i_arr = []
    mode = None
    if ((target_variance is not None) and
        (target_mean is not None)):
        mode = 'plot_both'
        i_arr = [0, 1]
    elif (target_variance is not None):
        mode = 'plot_variance'
        i_arr = [0]
    elif (target_mean is not None):
        mode = 'plot_mean'
        i_arr = [1]
    else:
        raise ValueError(f"You should provide target function for plotting") 
    y_label_name_arr = [r'$\mathbb{V}[f(x)]$', r'$\mathbb{E}[f(x)]$']
    label_target_arr = ['target variance', 'target mean']
    target_arr = [target_variance, target_mean]
    prediction_arr = [prediction_variance, prediction_mean]
    label_prediction_arr = [comparison_method + ' prediction variance', comparison_method + ' prediction mean']
    title_arr = ['Variance function', 'Mean function']
    for i in i_arr:
        if mode == 'plot_both':
            plt.subplot(1, 2, i + 1)
        plt.xlabel(r'$x$')
        plt.ylabel(y_label_name_arr[i])
        plt.plot(grid, target_arr[i], label = label_target_arr[i])
        if prediction_arr[i] is not None:
            plt.plot(grid, prediction_arr[i], label = label_prediction_arr[i])
        plt.title(title_arr[i])
        plt.grid(True)
        plt.legend()
