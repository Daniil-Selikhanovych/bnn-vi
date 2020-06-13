# bnn-vi

This repository contains experiments based on the paper by [Y. K. Foong et al, 2018](https://arxiv.org/pdf/1909.00719.pdf). The authors of the paper presented results about Variational Inference in Bayesian neural networks for the fundamental problem of uncertainty quantification. They showed some problems with the most popular approximate inference families: Mean Field Variational Inference and Monte Carlo Dropout.

## What's inside?

Here you can find a brief description of the experiments implemented in this repository.

### Regression 1D dataset

<p align="center">
  <img width="500" alt="Regression dataset" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/dataset_and_screenshots/reg_gt.jpg?raw=true">
</p>

We start from our synthetic 1D regression dataset:
$$
\mathcal{D}_{\text{reg}} = \{(x_k, y_k) \}_{k=1}^{100},
$$
$$
\pi_k \sim \text{Be} (0.5), \quad
x_k \sim \mathcal{N}\left(\frac{(-1)^{\pi_k}}{\sqrt{2}}, 0.1^2\right),
$$
$$
y_k \sim f(x_k) + \varepsilon, \quad \quad \varepsilon \sim \mathcal{N}(0, 0.1^2),
$$
$$
f(x) = \sin 12x + 0.66 \cos 25x + 3.
$$

Random **seed** was $42$ for both $x_{k}$ and $y_{k}$. Unfortunately, the authors do not write anything about the regression function and the data generation process, so we made this choice in our experiments on our own. The task is to recover the unknown regression function and its uncertainty using 4 methods: Gaussian Processes (GP), Hamiltonian Monte Carlo (HMC), Variational inference with 2 approximate families: Mean Field Variational Inference (MVFI) and Monte Carlo Dropout (MCDO).

### Baseline 1: Gaussian Processes

<p align="center">
  <img width="500" alt="Gaussian Processes" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/GP/reg_gp.jpg?raw=true">
</p>

In this experiment we learn GP using **GPflow** library. We trained GP with **maxiter = 100** with the following parameters: kernel **Matern52** with known variance **variance = 0.01** and lengthscales **lengthscales = 0.3**. One of its most natural properties is the increase in uncertainty (standard deviation) in the regions between training data points.

### Bayesian neural network architecture and setup 

For all experiments we used a multi-layer fully-connected ReLU network with 50 hidden units on each hidden layer. We assume that the conditional distribution of target is $\mathcal{N}(y_{k}, \sigma^{2})$, where $\sigma = 0.1$ is constant for all observations and $y_k$ is the value provided as ground-truth. The prior for mean is set to zero for all parameters. The standard deviation of biases is set to one. Suppose, that there is layer $i$ with $N_{in, i}$ inputs and $N_{out, i}$. For each layer $i \in \overline{1, 10}$ we used $\sigma_{w, depth}/\sqrt{N_{in, i}}$ for the prior standard deviation of each weight with $\sigma_{w} =\{\sigma_{first},3,2.25,2,2,1.9,1.75,1.75,1.7,1.65\}$ (array $\sigma_{w}$ denotes $\sigma_{w, depth}$ for each depth of BNN). We will describe our $\sigma_{first}$ choice for each experiment. In original paper authors use $\sigma_{first} = 4$. According to [Tomczak et al., 2018] we initalize set biases mean to zero and standard deviation to one, weights standard deviations are set to $10^{-5}$ and their means are independent samples from $\mathcal{N}\left(1, \frac{1}{\sqrt{4N_{out, i}}}\right)$ for the layer $i \in \overline{1, 10}$.

### Baseline 2: Hamiltonian Monte Carlo 

<p align="center">
  <img width="500" alt="Hamiltonian Monte Carlo" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/regression/1layer_reg_pool.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Deterministic neural network" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/regression/1layer_reg_det.jpg?raw=true">
</p>

We use **BNN** with 1 layer and $\sigma_{first} = 10$. We train **NUTS** with 5 parallel chains of MCMC and 300 samples from each chain for the distribution estimation and 300 samples for warmup. Result predicition is based on ensemble of these 1500 models of NUTS generated sets of weights. In Pyro we set random seed as **pyro.set_rng_seed(1)** before BNN initialization. We compare our result with training simple deterministic of neural network with the same architecture. For this NN we used **Adam** optimizer with $lr = 10^{-3}$, MSE loss and **num epochs = 1000**. We see that deterministic network tends to fit data worse than Bayesian and the Bayesian setting gives smoother results. Results are shown in the 2 figures: the top figure for the NUTS method and the bottom figure for the deterministic neural network. It can be seen, that the uncertainty is higher in the region between two clusters.

### Variational inference: MFVI

#### Custom model 

In this experiment we train 1 layer BNN with the same setup using MFVI approximation family with **ELBO** loss. We estimate ELBO with only one sample (**num particles = 1**), as we discovered that it speeds up the convergence, together with reducing computation time per sample. We trained it using **SVI** class in **Pyro** library with Adam optimizer, $lr = 10^{-3}$ for **num epochs = 30000** and batch size equal to the whole dataset size. We set random seed as **pyro.set_rng_seed(1)** before training process. Firstly, we show results for different prior choice $\sigma_{first} = \{0.1, 1, 10\}$ from top to bottom. We can see, that the plain approximator is sensitive to the target’s prior scale.

<p align="center">
  <img width="500" alt="Prior weight standard deviation 0.1" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/1layer_reg_VI_std0.1.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 1" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/1layer_reg_VI_std1.0.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 10" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/1layer_reg_VI_std10.jpg?raw=true">
</p>

We see that optimization process can be unsuccessful for very small prior weight standard deviation: training process is smooth without steps, which indicates only training for uncertainty, but not for the source data in term of mean prediction. The first picture shows that the neural network cannot describe the data well, although the optimization process has converged. We demonstrate loss graphs for the cases $\sigma_{first} = 0.1$ and $\sigma_{first} = 10$ from top to bottom. 

<p align="center">
  <img width="500" alt="Loss for prior weight standard deviation 0.1" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/losses/1layer_reg_VI_log_loss_bad.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Loss for prior weight standard deviation 10" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/losses/1layer_reg_VI_log_loss.jpg?raw=true">
</p>

#### Model with Local Reparametrization Trick

We emphasize, that Local Reparametrization Trick [Kingma et al., 2015] was used in the original paper. It is believed to simplify the training process due to smaller covariances between the gradients in one batch and, what is more im-
portant, it makes computations more efficient. We implemented this method by scratch. We demonstrate our results for different prior choice $\sigma_{first} = \{0.1, 1, 10\}$ (from top to bottom) with the same setup as for the custom model.   

<p align="center">
  <img width="500" alt="Prior weight standard deviation 0.1 with trick" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/1layer_reg_VItrick_std0.1.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 1 with trick" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/1layer_reg_VItrick_std1.0.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 10 with trick" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/1layer_reg_VItrick_std10.jpg?raw=true">
</p>

We see that Local Reparametrization Trick poses independence from the prior scale selection. Even though it can make the optimization process more robust, it definitely gives us less control. We compare deeper models: consider the same architecture for the custom model and Local Reparametrization Trick, but with 2 layers. The results are presented in the following figures, top figure corresponds to Local Reparametrization Trick with $\sigma_{first} = 1$ and the bottom figure corresponds to the custom model with $\sigma_{first} = 10$.

<p align="center">
  <img width="500" alt="Prior weight standard deviation 1 with trick for 2 layers" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/2layer_reg_VItrick_std1.0.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 10 for 2 layers" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/2layer_reg_VI_std10.jpg?raw=true">
</p>

Even though it is easier for them to fit the data, there is no significant change in the uncertainty estimation. We emphasize that the usual notion of stacking layers to boost the models complexity doesn’t apply here, so we should keep looking for other approximation techniques.

### Classification 2D dataset and HMC 

Our synthetic 2D classification dataset via regression is the following:
$$
\mathcal{D}_{\text{class}} = \{(x_k, y_k) \}_{k=1}^{100},
$$
$$
\pi_k \sim \text{Be} (0.5), \quad
x_k \sim \mathcal{N}\left(\frac{(-1)^{\pi_k}}{\sqrt{2}}\left(1, 1\right)^{T}, 0.2^{2}I_{2 \times 2}\right), 
$$
$$
y_k \sim f(x_k) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 0.1^2),
$$
$$
f(x) = (-1)^{\pi_k}.
$$
We trained 2 layers BNN with the same setup for NUTS as for 1D regression task. We end up with the posterior depicted in the following figures:

<p align="center">
  <img width="500" alt="Classification HMC pool" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/classification/1layer_2d_pool.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Classification HMC contourf" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/classification/1layer_2d_pool_contourf.jpg?raw=true">
</p>
Unfortunately, we didn't obtain good results for this case. The overall scale of the dataset is the same, as in the regression task, thus it seems possible for the Bayesian model to fit the data given correct hyperparameters, but not for this set of hyperparameters.

### Regression 2D dataset

We present our synthetic 2D regression dataset. Consider two clusters of points $(x,y)$ with centers in $\left(-\frac{1}{\sqrt{2}},-\frac{1}{\sqrt{2}}\right)$ and $\left(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}\right)$ with 100 points in each cluster drawn from normal distributions with standard deviation equal $0.1$. This will be the input variables for our model. The target is simply the evaluation of $f(x,y) = \sin(12 x y) + 0.66 \cos(25(x+y)) + \exp(x-y) + \varepsilon, \varepsilon \sim \mathcal{N}(0,1)$ at these points. Our objective is the uncertainty (standard deviation or variance) predicted by the model for the set $[-2,2]\times[-2,2]$. 

#### Variance prediction from model and losses from scratch

In these experiments we used our own implementation of BNNs with MFVI and MCDO approximation families based only on **PyTorch** framework. We had to implement BNNs and losses from scratch.

#### Model with unknown conditional variance

This model is analogous to the original with the only distinction, that the conditional variance is unknown and non constant ($\sigma_{noise} = 0.1$ in previous examples). Namely, we assume that the conditional distribution of target is given by: $y|\mathbf{x},\mathbf{w} \sim \mathcal{N}(f^{\mu}_{\mathbf{w}}(\mathbf{x}),f^{\sigma^2}_{\mathbf{w}}(\mathbf{x}))$, i.e. the variance is predicted by the network. In this model the uncertainty is measured by $\mathbb{E}_{\mathbf{w}}[f^{\sigma^2}_{\mathbf{w}}(\mathbf{x})]$ and not by $\mathbb{V}_{\mathbf{w}}(f^{\mu}_{\mathbf{w}}(\mathbf{x}))$ as in the previous cases. In that case BNN has 2 outputs: first is the mean of normal distribution $output_{BNN}^{(1)} = f^{\mu}_{\mathbf{w}}(\mathbf{x})$ and the second is $output_{BNN}^{(2)}$. We will describe how $output_{BNN}^{(2)}$ is connected with $\mathbb{V}_{\mathbf{w}}(f^{\mu}_{\mathbf{w}}(\mathbf{x}))$ in the section **Custom ELBO loss**.

#### Details of BNNs implementation from scratch

We used the following formulas for forward method in BNNs when we sample weights for current layer $i$:
$$
W_{i} = \mu_{W, i} + \log\left(1 + \varepsilon + \exp\left(\rho_{W, i}\right)\right) \odot \operatorname{noise}_{1}, 
$$
$$
b_{i} = \mu_{b, i} + \log\left(1 + \varepsilon + \exp\left(\rho_{b, i}\right)\right) \odot \operatorname{noise}_{2},
$$
$$
\varepsilon = 10^{-4},
$$

where $\mu_{W, i}$ is an matrix with size $N_{i, out} \times N_{i, in}$ of learned means for elements in weight matrix $W_{i}$; $\rho_{W, i}$ is an matrix of learned parameters with size $N_{i, out} \times N_{i, in}$ that determines standard deviations for weight matrix $W_{i}$ (matrix $\log\left(1 + \varepsilon + \exp\left(\rho_{W, i}\right)\right)$ is obtained element-wise from the matrix $\rho_{W, i}$); $\operatorname{noise}_{1}$ is an matrix with size $N_{i, out} \times N_{i, in}$ such as all elements of $\operatorname{noise}_{1}$ are sampled from $\mathcal{N}(0, 1)$ independently. The same notation is used for $\mu_{b, i}, \rho_{b, i}, \operatorname{noise}_{2}$: $\mu_{b, i}$ is an vector with size $N_{i, out}$ of learned means for elements in bias vector $b_{i}$; $\rho_{b, i}$ is an vector of learned parameters with size $N_{i, out}$ that determines standard deviations for bias vector $b_{i}$ (vector $\log\left(1 + \varepsilon + \exp\left(\rho_{W, i}\right)\right)$ is obtained element-wise from the vector $\rho_{b, i}$); $\operatorname{noise}_{2}$ is an vector with size $N_{i, out}$ such as all elements of $\operatorname{noise}_{2}$ are sampled from $\mathcal{N}(0, 1)$ independently. 

During testing for the model with constant conditional variance we calculate std for BNN output as Monte Carlo estimation using 128 samples of numerical stds for $output_{BNN}^{(1)}$ according to [Y. K. Foong et al, 2018]. For the model with unknown conditional variance (also during testing) we calculate std for BNN output as Monte Carlo estimation using 128 samples of $\sqrt{\mathbb{V}_{\mathbf{w}}(f^{\mu}_{\mathbf{w}}(\mathbf{x}))}$. Also for testing we use BNNs mean output as Monte Carlo estimation using 128 samples of $output_{BNN}^{(1)}$. During training these objectives were estimated using 32 Monte Carlo samples according to [Y. K. Foong et al, 2018].

#### ELBO loss from scratch
We implemented an ELBO loss with batches calculation:
$$
\mathcal{L}(\phi)=\sum_{n=1}^{N} \mathbb{E}_{q_{\phi}}\left[\log p\left(y_{n} | \mathbf{x}_{n}, f_{\theta}\right)\right]-\mathrm{KL}\left[q_{\phi}(\theta) \| p(\theta)\right]
$$
can be rewritten as
$$
\sum\limits_{i = 1}^{\operatorname{num \; batches}} \left(\mathbb{E}_{q_{\phi}}\left[\sum\limits_{j = 1}^{\operatorname{batch}_{i}}\left[\log p\left(y_{j, \operatorname{batch}_{i}} | \mathbf{x}_{j, \operatorname{batch}_{i}}, f_{\theta}\right)\right]\right]-\frac{\operatorname{batch size}_{i}}{N}\mathrm{KL}\left[q_{\phi}(\theta) \| p(\theta)\right]\right)
$$

where $N = \sum\limits_{i = 1}^{\operatorname{num \; batches}}\operatorname{batch size}_{i}$ is a size of train dataset. 

The expectation for log-density part in ELBO loss was estimated using 32 Monte Carlo samples during training according to [Y. K. Foong et al, 2018]. For variance prediction models we used a trick for numerical stability. When calculating log-density Monte-Carlo estimation we implement the following formulas to obtain std prediction $\sqrt{\mathbb{V}_{\mathbf{w}}(f^{\mu}_{\mathbf{w}}(\mathbf{x}))}$ using the second BNNs output $output_{BNN}^{(2)}$:
$$
\sqrt{\mathbb{V}_{\mathbf{w}}(f^{\mu}_{\mathbf{w}}(\mathbf{x}))} = \begin{cases}
\log\left(1 + \varepsilon + \exp\left(output_{BNN}^{(2)}\right)\right) \quad \operatorname{if } output_{BNN}^{(2)} \leq 60, \\
output_{BNN}^{(2)} \quad \operatorname{else}
\end{cases}
$$

We used $\varepsilon = 10^{-4}$ in all experiments. So, for $x > 60$ we rely on $\log(1 + \exp(x)) \approx x$. 

#### MFVI using ELBO loss

We trained BNNs on the above regression 2D dataset for 1000 epochs with Adam optimizer with $lr = 10^{-2}$.  We present results for training BNNs with 1 and 2 layers for 2 modes: with constant conditional variance $\sigma_{noise} = 0.1$ in the model of data $y_k \sim f(x_k) + \varepsilon, \varepsilon \sim \mathcal{N}(0, \sigma_{noise}^2)$ and with unknown conditional variance in the model of data $y|\mathbf{x},\mathbf{w} \sim \mathcal{N}(f^{\mu}_{\mathbf{w}}(\mathbf{x}),f^{\sigma^2}_{\mathbf{w}}(\mathbf{x}))$. We used stds estimation that was described in the section **Details of BNNs implementation from scratch**.  Firstly, we show results for 1 layer BNNs, top figure corresponds to the model with constant conditional variance and the bottom figure corresponds to the model with unknown conditional variance.

<p align="center">
  <img width="500" alt="1 HL BNN with constant conditional variance" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/std_prediction/1HL_MFVI%20without_variance.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="1 HL BNN with unknown conditional variance" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/std_prediction/1HL_MFVI_with_variance.jpeg?raw=true">
</p>

The same results for 2 layers BNNs are presented in the following figures:

<p align="center">
  <img width="500" alt="2 HL BNN with constant conditional variance" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/std_prediction/2HL_MFVI_without_variance.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="2 HL BNN with unknown conditional variance" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/std_prediction/2HL_MFVI_with_variance.jpeg?raw=true">
</p>

We see that for MFVI with constant conditional variance the in-between clusters uncertainty is lower. However, we can not see that the uncertainty function is convex: convexity is preserved only on some line segments as Theorem 1 in [Y. K. Foong et al, 2018] claims. We also see that for 2 hidden layers network the convexity is preserved over the smaller area, compared to 1 hidden layer network. This gives us an intuition that increasing the number of hidden layers may reduce convexity of uncertainty function.

#### MCDO loss from scratch
It was shown in [Gal & Ghahramani, 2016] that maximizing ELBO with MCDO family is equivalent to minimizing
$$
||y-\mathbb{E}f(X)||_2^2+\lambda \sum_{i=1}^{k} (||W_i||_2^2+||b_i||^2_2),
$$

where $y$ is the vector of all target values in the training dataset, $X$ is the matrix of input variables, $\mathbb{E}f(X)$ is the vector of expectations of BNN predictions, $k$ is the number of fully-connected layers, $W_i$ and $b_i$ are weights and biases of the $i'th$ layer, for a properly chosen $\lambda$. Also it was shown that in order to treat Dropout as Bayesian inference we should choose $\lambda$ by the formula:
$$
\lambda = \frac{pl^2}{2N\tau},
$$

where $p$ is the dropout probability, $l^2$ is the reciprocal of the prior variance for the weigths in the first fully-connected layer, $N$ is the size of the training data and $\tau$ is the conditional variance of $y|\mathbf{x},\mathbf{w}$. According to [Y. K. Foong et al, 2018] we used $p = 0.05$. Also we used $\tau = 1$ and according to notation in [Gal, 2016] $l_{i} = \left(\sigma_{w, depth}/\sqrt{N_{in, i}}\right)^{-1} = \sqrt{N_{in, i}}/\sigma_{w, depth}$. 

#### MCDO results

In this experiment we train BNNs on the regression 2D dataset for 1000 epochs with Adam optimizer with $lr = 10^{-3}$ using MCDO approximation family with MCDO loss discussed above. We used stds estimation that was described in the section **Details of BNNs implementation from scratch**.  We show results for BNN with 1 and 2 layers from top to bottom.

<p align="center">
  <img width="500" alt="1 HL MCDO BNN" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/std_prediction/1HL_MCDO.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="2 HL MCDO BNN" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/std_prediction/2HL_MCDO.jpeg?raw=true">
</p>

We can observe that convexity is preserved in both cases.

#### Baseline for regression 2D dataset: Gaussian Processes

Now we show result for regression task in the section **Regression 2D dataset** using Gaussian Processes from **from sklearn.gaussian_process**. We used **GaussianProcessRegressor** with $ConstantKernel \times Matern$ as a kernel and **n_restarts_optimizer=9** parameter. Parameters for $Matern$ kernel were the following: $\nu = 20, \operatorname{lengthscale \; bounds} = (10^{-1}, 10^{10})$. Parameters for $ConstantKernel$ kernel were the following: $\operatorname{constant \; value} = 10, \operatorname{constant \; value \; bounds} = (10^{-3}, 10^{3})$.

<p align="center">
  <img width="500" alt="Gaussian Processes 2D" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/GP/Gaussian_Process.png?raw=true">
</p>

We see that the level lines of uncertainty are not convex and display the desirable property: the further away we are from the training points the less confident we are in the prediction. There is no problems with in-between uncertainty or convexity in general.

### Learning BNNs for given mean and variance functions

In this section we are going to the hypothesis that the poor uncertainty prediction is due to improper architecture of neural networks or not. For this task we will try to fit BNNs with certain functions, which play the role of mean and variance functions.

#### Mean/variance functions and loss

We consider functions $h(x)=\frac{1}{\pi \gamma} \frac{\gamma^{2}}{x^{2}+\gamma^{2}}$ with $\gamma = 0.3$ and $g(x)=\sin (2 x)+0.66 \cos (7 x)+\exp \left(x^{2}\right)$ on the segment $[-1, 1]$. We will fit different networks using the
following $l_2$ loss function:
$$
\left\|\mathbb{E}_{\mathbf{w}}\left[f_{\mathbf{w}}(x)\right]-g(x)\right\|_{2}^{2}+\left\|\mathbb{V}_{\mathbf{w}}\left[f_{\mathbf{w}}(x)\right]-h(x)\right\|_{2}^{2}.
$$

#### Train dataset

Our train dataset consists of $N = 40$ points according to [Y. K. Foong et al, 2018] with values of the functions $f(x), g(x)$ in these points. These points were randomly sampled from the uniform grid of 1000 points on the segment $[-1, 1]$.

#### MFVI training

We train BNNs for 6000 epochs with Adam optimizer, $lr = 10^{-3}$ using MFVI approximation family with $l_2$ loss from the section **Mean/variance functions and loss**. We present results for training BNNs with 1 and 2 layers for the mode with constant conditional variance. We used stds and means estimations that were described in the section **Details of BNNs implementation from scratch**. We show results for 1 layer and 2 layers BNNs, top 2 figures correspond to BNN with 1 layer and the bottom 2 figures correspond to BNN with 2 layers. $x$ axis means indexes of points in grid on the segment $[-1, 1]$.

<p align="center">
  <img width="500" alt="1 HL MFVI BNN mean prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/l2_loss/1HL%20MFVI%20fixed%20var%20mean%20predictions.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="1 HL MFVI BNN var prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/l2_loss/1HL%20MFVI%20fixed%20var%20variance%20prediction.jpeg?raw=true">
</p>

<p align="center">
  <img width="500" alt="2 HL MFVI BNN mean prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/l2_loss/2HL%20MFVI%20fixed%20var%20mean%20predictions.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="2 HL MFVI BNN var prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/l2_loss/2HL%20MFVI%20fixed%20var%20variance%20prediction.jpeg?raw=true">
</p>

We see that 1 layer MFVI BNN predicts mean relatively good, but not variance. This means that one layer network architecture is not powerful enough. However for 2 layer MFVI BNN both mean and variance the predictions are quite good. This confirms the Theorem 3 in [Y. K. Foong et al, 2018].

#### MCDO training

The same conclusion can be done for MCDO approximation family. We also trained MCDO BNNs for 6000 epochs with Adam optimizer, $lr = 10^{-3}$ and $l_2$ loss from the section **Mean/variance functions and loss**. We used stds and means estimations that were described in the section **Details of BNNs implementation from scratch** for the mode with constant conditional variance. We show results for 1 layer and 2 layers BNNs, top 2 figures correspond to BNN with 1 layer and the bottom 2 figures correspond to BNN with 2 layers. $x$ axis means indexes of points in grid on the segment $[-1, 1]$.

<p align="center">
  <img width="500" alt="1 HL MCDO BNN mean prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/l2_loss/1HL%20MCDO%20mean%20predictions.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="1 HL MCDO BNN var prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/l2_loss/1HL%20MCDO%20variance%20prediction.jpeg?raw=true">
</p>

<p align="center">
  <img width="500" alt="2 HL MCDO BNN mean prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/l2_loss/2HL%20MCDO%20mean%20predictions.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="2 HL MCDO BNN var prediction" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/l2_loss/2HL%20MCDO%20variance%20prediction.jpeg?raw=true">
</p>

## Requirements

We have run the experiments on Linux. The versions are given in brackets. The following packages are used in the implementation:
* [PyTorch (1.4.0)](https://pytorch.org/get-started/locally/)
* [NumPy (1.17.3)](https://numpy.org/)
* [scikit-learn (0.22.1)](https://scikit-learn.org/stable/)
* [matplotlib (3.1.2)](https://matplotlib.org/)
* [tqdm (4.39.0)](https://github.com/tqdm/tqdm)
* [Pyro (1.3.1)](https://pyro.ai/)
* [GPflow (2.0.4)](https://www.gpflow.org/)
* [TensorFlow (2.1.0) as dependency for GPflow](https://www.tensorflow.org/)


You can use [`pip`](https://pip.pypa.io/en/stable/) or [`conda`](https://docs.conda.io/en/latest/) to install them. 

## Contents

All the experiments can be found in the underlying notebooks:

| Notebook      | Description |
|-----------|------------|
|[notebooks/develop.ipynb](https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/notebooks/develop.ipynb) | **HMC, Local Reparametrization Trick, prior tuning:** experiments with HMC, Local Reparametrization Trick and prior tuning for MFVI BNNs.|
|[notebooks/Experiments.ipynb](https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/notebooks/Experiments.ipynb) | **2D regression, MCDO and MFVI from scratch:** experiments with our own implementation of losses and training BNNs using **PyTorch**.
|[notebooks/BNN_start.ipynb](https://github.com/Daniil-Selikhanovych/neural-ot/blob/master/notebooks/generative_modeling.ipynb)| **MNIST, GPflow and MVFI BNNs using Pyro**: experiments with **MNIST** dataset, 1D regression task with **GPflow** and MFVI BNNs using Pyro primitives and pyro.nn.Module. 

For convenience, we have also implemented a framework and located it correspondingly in [bnn-vi/bnn-vi](https://github.com/Daniil-Selikhanovych/bnn-vi/tree/master/bnn_vi), [bnn-vi/notebooks/api](https://github.com/Daniil-Selikhanovych/bnn-vi/tree/master/notebooks/api) and [bnn-vi/notebooks/Models_and_losses.py](https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/notebooks/Models_and_losses.py).

## Our team

At the moment we are *Skoltech DS MSc, 2019-2021* students.
* Artemenkov Aleksandr 
* Karpikov Igor
* Selikhanovych Daniil
