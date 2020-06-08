import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroParam, PyroSample

std_weights_default = [4., 3., 2.25, 2, 2, 1.9, 1.75, 1.75, 1.7, 1.65]
std_bias_default = 1.
mean_prior_default = 0.
hidden_features_default = 50
n_layers_default = 1
device_default = 'cuda:0'

num_samples_default = 128

def l2_loss(mean_true, var_true, mean_pred_mc, var_pred_mc):
    loss = nn.MSELoss(reduction = 'sum')
    return loss(mean_true, mean_pred_mc) + loss(mean_pred_mc, var_pred_mc)

class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features, 
                 std_weights = std_weights_default,
                 std_bias = std_bias_default,
                 mean_prior = mean_prior_default,
                 hidden_features = hidden_features_default, 
                 n_layers = n_layers_default, 
                 device = device_default):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_layers = n_layers
        self.device = device
        
        if (self.n_layers > len(std_weights) or self.n_layers < 1):
            raise ValueError(f"Num layers should be between 1 and {len(std_weights)}!")
        
        pipe = [PyroModule[nn.Linear](in_features, hidden_features),
                PyroModule[nn.ReLU]()]
        for i in range(n_layers - 1):
            pipe += [PyroModule[nn.Linear](hidden_features, hidden_features),
                     PyroModule[nn.ReLU]()]
            
        pipe += [PyroModule[nn.Linear](hidden_features, out_features)]
        self.seq = PyroModule[nn.Sequential](*pipe)
        # See the paper
        std_weights = std_weights[::-1][:n_layers+1][::-1]
        k = -1
        for i in range(len(self.seq)):
            if 'linear' in type(self.seq[i]).__name__.lower():
                k += 1
                out_size, in_size = self.seq[i].weight.shape 
                # We can't specify the device explicitly, thus using this hack
                self.seq[i].bias = PyroSample(dist.Normal(torch.tensor(mean_prior, device = device), std_bias,
                                                          validate_args = False).expand([out_size]).to_event(1))
                self.seq[i].weight = PyroSample(dist.Normal(torch.tensor(mean_prior, device = device),
                                                            std_weights[k]/hidden_features**0.5, 
                                                            validate_args = False).expand([out_size, in_size]).to_event(2))
        
    def forward(self, x, y = None):
        y_pr = self.seq(x)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(y_pr, 0.1).to_event(1), obs = y)
        return y_pr.detach()
        
class BayesianLinearMFVI(BayesianLinear):
    def __init__(self, in_features, out_features, 
                       std_weights = std_weights_default,
                       std_bias = std_bias_default,
                       mean_prior = mean_prior_default,
                       hidden_features = hidden_features_default, 
                       n_layers = n_layers_default, 
                       device = device_default):
        super().__init__(in_features, out_features, std_weights, std_bias, mean_prior, hidden_features, n_layers, device)
        self.guide = AutoDiagonalNormal(self)      
        
    def summary(self, samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                        "mean": torch.mean(v, 0),
                        "var": torch.var(v, 0)
                        }
        return site_stats

    def custom_l2_loss(self, model, guide, *args, **kwargs):
        # run the guide and trace its execution
        X, num_samples, pred_X_mean, pred_X_var = args

        predictive = Predictive(self, guide = self.guide, 
                                num_samples = num_samples,
                                return_sites = ("obs", "_RETURN"))
        samples = predictive(X)
        pred_summary = self.summary(samples)
        mu = pred_summary["_RETURN"]
        y = pred_summary["obs"]
        mu_mean = mu["mean"]
        mu_std = mu["std"]
        
        return l2_loss(pred_X_mean, pred_X_var, mu_mean, mu_std)
   
    def train_VI_l2(self, x_train, gp, num_epoch = 80000,
                    num_samples = num_samples_default,
                    lr = 1e-2, every_epoch_to_print = 1000):     
        optimizer = Adam({"lr": lr})
        svi = SVI(self, self.guide, optimizer, 
                      loss = self.custom_l2_loss)
        mean_gp, var_gp = np.array(gp.predict_f(x_train))  
        mean_gp_torch = torch.Tensor(mean_gp[:, 0]).to(self.device)
        var_gp_torch = torch.Tensor(var_gp[:, 0]).to(self.device)
        x_train_torch = torch.Tensor(x_train).to(self.device)
        pyro.clear_param_store()
        loss_arr = []
        for j in range(num_epoch):
            # calculate the loss and take a gradient step
            loss = svi.step(x_train_torch, num_samples, mean_gp_torch, var_gp_torch)
            loss_arr.append(loss)
            if j % every_epoch_to_print == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))
        return loss_arr
        
    def train_VI_ELBO(self, x_train, y_train, num_epoch = 80000, 
                      lr = 1e-2, every_epoch_to_print = 1000):
        optimizer = Adam({"lr": lr})
        svi = SVI(self, self.guide, optimizer, 
                      loss = Trace_ELBO())
        pyro.clear_param_store()
        loss_arr = []
        for j in range(num_epoch):
            # calculate the loss and take a gradient step
            loss = svi.step(x_train, y_train)
            loss_arr.append(loss)
            if j % every_epoch_to_print == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))
        return loss_arr


class BNN_1_layer(PyroModule):
    def __init__(self, in_features, num_hidden, learn_var = False):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](in_features, num_hidden)
        self.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([num_hidden, in_features]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 10.).expand([num_hidden]).to_event(1))
        self.learn_var = learn_var
        if not self.learn_var:
            self.linear2 = PyroModule[nn.Linear](num_hidden, 2)
            self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([2, num_hidden]).to_event(2))
            self.linear2.bias = PyroSample(dist.Normal(0., 10.).expand([2]).to_event(1))
        else:
            self.linear2 = PyroModule[nn.Linear](num_hidden, 1)
            self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([1, num_hidden]).to_event(2))
            self.linear2.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

        self.guide = AutoDiagonalNormal(self)

    def forward(self, x, y = None):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        if not self.learn_var:
            mean = out[:, 0]
            std = F.softplus(out[:, 1])
        else:
            mean = out
            sigma = pyro.sample("sigma", dist.Normal(0., 10.))
            std = F.softplus(sigma)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, std), obs=y)
        return out

    def train(self, x_train, y_train, num_epoch = 80000, 
              lr = 1e-2, every_epoch_to_print = 1000):
        optimizer = Adam({"lr": lr})
        svi = SVI(self, self.guide, optimizer, 
                  loss = Trace_ELBO())
        pyro.clear_param_store()
        loss_arr = []
        for j in range(num_epoch):
            # calculate the loss and take a gradient step
            loss = svi.step(x_train, y_train)
            loss_arr.append(loss)
            if j % every_epoch_to_print == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))
        return loss_arr

    def summary(self, samples):
        site_stats = {}
        for k, v in samples.items():
            if (k == "_RETURN" and (not self.learn_var)):
                site_stats[k] = {
                        "mean": torch.mean(v[:, :, 0], 0),
                        "std": torch.mean(F.softplus(v[:, :, 1]), 0)
                }
            else:
                site_stats[k] = {
                        "mean": torch.mean(v, 0),
                        "std": torch.std(v, 0)
                        }
        return site_stats

    def sample(self, x_test, num_samples = 128):
        self.guide.requires_grad_(False)

        predictive = Predictive(self, guide = self.guide, 
                                num_samples = num_samples,
                                return_sites = ("obs", "_RETURN"))
        samples = predictive(x_test)
        pred_summary = self.summary(samples)
        mu = pred_summary["_RETURN"]
        y = pred_summary["obs"]
        mu_mean = mu["mean"]
        mu_std = mu["std"]
        mu_var = mu_std.pow(2)
        y_mean = y["mean"]
        y_std = y["std"]
        y_var = y_std.pow(2)
        
        return mu_mean, mu_var, y_mean, y_var
        
class BNN_2_layer(PyroModule):
    def __init__(self, in_features, num_hidden, learn_var = False):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](in_features, num_hidden)
        self.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([num_hidden, in_features]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 10.).expand([num_hidden]).to_event(1))
        self.learn_var = learn_var
        self.linear2 = PyroModule[nn.Linear](num_hidden, num_hidden)
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([num_hidden, num_hidden]).to_event(2))
        self.linear2.bias = PyroSample(dist.Normal(0., 10.).expand([num_hidden]).to_event(1))
        if not self.learn_var:
            self.linear3 = PyroModule[nn.Linear](num_hidden, 2)
            self.linear3.weight = PyroSample(dist.Normal(0., 1.).expand([2, num_hidden]).to_event(2))
            self.linear3.bias = PyroSample(dist.Normal(0., 10.).expand([2]).to_event(1))
        else:
            self.linear3 = PyroModule[nn.Linear](num_hidden, 1)
            self.linear3.weight = PyroSample(dist.Normal(0., 1.).expand([1, num_hidden]).to_event(2))
            self.linear3.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

        self.guide = AutoDiagonalNormal(self)

    def forward(self, x, y = None):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        if not self.learn_var:
            mean = out[:, 0]
            std = F.softplus(out[:, 1])
        else:
            mean = out
            sigma = pyro.sample("sigma", dist.Normal(0., 10.))
            std = F.softplus(sigma)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, std), obs=y)
        return out

    def train(self, x_train, y_train, num_epoch = 80000, 
              lr = 1e-2, every_epoch_to_print = 1000):
        optimizer = Adam({"lr": lr})
        svi = SVI(self, self.guide, optimizer, 
                  loss = Trace_ELBO())
        pyro.clear_param_store()
        loss_arr = []
        for j in range(num_epoch):
            # calculate the loss and take a gradient step
            loss = svi.step(x_train, y_train)
            loss_arr.append(loss)
            if j % every_epoch_to_print == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))
        return loss_arr

    def summary(self, samples):
        site_stats = {}
        for k, v in samples.items():
            if (k == "_RETURN" and (not self.learn_var)):
                site_stats[k] = {
                        "mean": torch.mean(v[:, :, 0], 0),
                        "std": torch.mean(F.softplus(v[:, :, 1]), 0)
                }
            else:
                site_stats[k] = {
                        "mean": torch.mean(v, 0),
                        "std": torch.std(v, 0)
                        }
        return site_stats

    def sample(self, x_test, num_samples = 128):
        self.guide.requires_grad_(False)

        predictive = Predictive(self, guide = self.guide, 
                                num_samples = num_samples,
                                return_sites = ("obs", "_RETURN"))
        samples = predictive(x_test)
        pred_summary = self.summary(samples)
        mu = pred_summary["_RETURN"]
        y = pred_summary["obs"]
        mu_mean = mu["mean"]
        mu_std = mu["std"]
        mu_var = mu_std.pow(2)
        y_mean = y["mean"]
        y_std = y["std"]
        y_var = y_std.pow(2)
