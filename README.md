# bnn-vi

This repository contains experiments based on the paper by [Y. K. Foong et al, 2018](https://arxiv.org/pdf/1909.00719.pdf). The authors of the paper presented results about Variational Inference in Bayesian neural networks for the fundamental problem of uncertainty quantification. They showed some problems with the most popular approximate inference families: Mean Field Variational Inference and Monte Carlo Dropout.

## What's inside?

Here you can find a brief description of the experiments implemented in this repository.

### Regression 1D dataset

<p align="center">
  <img width="500" alt="Regression dataset" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/dataset_and_screenshots/reg_gt.jpg?raw=true">
</p>

We start from our synthetic 1D regression dataset:
<p align="center"><img src="svgs/8f86f90c443777ca894f55a461621417.svg?invert_in_darkmode" align=middle width=152.4124305pt height=18.905967299999997pt/></p>
<p align="center"><img src="svgs/a2190a22d732b75371b3d14354ef3168.svg?invert_in_darkmode" align=middle width=294.53239859999996pt height=39.452455349999994pt/></p>
<p align="center"><img src="svgs/2cf469448692cd4102dbd9caa080accb.svg?invert_in_darkmode" align=middle width=252.83896604999998pt height=18.312383099999998pt/></p>
<p align="center"><img src="svgs/a40ce0326fc0e14a28a6fa565d2bec1e.svg?invert_in_darkmode" align=middle width=238.1847633pt height=16.438356pt/></p>

Random **seed** was <img src="svgs/df52bf7ea910ee5450181708854d700e.svg?invert_in_darkmode" align=middle width=16.438418699999993pt height=21.18721440000001pt/> for both <img src="svgs/c34d63da9854e1c53ba51c021bdf1fa4.svg?invert_in_darkmode" align=middle width=16.66101689999999pt height=14.15524440000002pt/> and <img src="svgs/787ac731e86c7e339f5efcc8bc8f2384.svg?invert_in_darkmode" align=middle width=15.325460699999988pt height=14.15524440000002pt/>. Unfortunately, the authors do not write anything about the regression function and the data generation process, so we made this choice in our experiments on our own. The task is to recover the unknown regression function and its uncertainty using 4 methods: Gaussian Processes (GP), Hamiltonian Monte Carlo (HMC), Variational inference with 2 approximate families: Mean Field Variational Inference (MVFI) and Monte Carlo Dropout (MCDO).

### Baseline 1: Gaussian Processes

<p align="center">
  <img width="500" alt="Gaussian Processes" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/GP/reg_gp.jpg?raw=true">
</p>

In this experiment we learn GP using **GPflow** library. We trained GP with **maxiter = 100** with the following parameters: kernel **Matern52** with known variance **variance = 0.01** and lengthscales **lengthscales = 0.3**. One of its most natural properties is the increase in uncertainty (standard deviation) in the regions between training data points.

### Bayesian neural network architecture and setup 

For all experiments we used a multi-layer fully-connected ReLU network with 50 hidden units on each hidden layer. We assume that the conditional distribution of target is <img src="svgs/c623394445fd8fe375d720da765b550e.svg?invert_in_darkmode" align=middle width=69.50588864999999pt height=26.76175259999998pt/>, where <img src="svgs/9dc641f7e4c5a868ff46da9e6d1b890e.svg?invert_in_darkmode" align=middle width=52.90515614999999pt height=21.18721440000001pt/> is constant for all observations and <img src="svgs/6217c8bb52c9d9f0adb29e37b52dad41.svg?invert_in_darkmode" align=middle width=15.325460699999988pt height=14.15524440000002pt/> is the value provided as ground-truth. The prior for mean is set to zero for all parameters. The standard deviation of biases is set to one. Suppose, that there is layer <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> with <img src="svgs/fbc032ecfc013b32deeb5daf5d95d8ee.svg?invert_in_darkmode" align=middle width=34.53972224999999pt height=22.465723500000017pt/> inputs and <img src="svgs/ebcd127f4dbb9a3ca52329f2b542ec12.svg?invert_in_darkmode" align=middle width=40.98934619999999pt height=22.465723500000017pt/>. For each layer <img src="svgs/7f5282dd648752cd11332df8685daf9c.svg?invert_in_darkmode" align=middle width=57.71787614999998pt height=26.447203200000008pt/> we used <img src="svgs/13ec49286ed7ed66cf277795fe4b1301.svg?invert_in_darkmode" align=middle width=116.47580009999999pt height=27.73529880000001pt/> for the prior standard deviation of each weight with <img src="svgs/8e3199a71ab092dace1235e984981d7c.svg?invert_in_darkmode" align=middle width=347.8989426pt height=24.65753399999998pt/> (array <img src="svgs/3728b49665b38c535d71e9d95909c80e.svg?invert_in_darkmode" align=middle width=19.21242839999999pt height=14.15524440000002pt/> denotes <img src="svgs/8ec3fd3d6643dd8805aeaee1a1e41ba8.svg?invert_in_darkmode" align=middle width=55.63483199999999pt height=14.15524440000002pt/> for each depth of BNN). We will describe our <img src="svgs/e0ba7ed8f07fe8606150f260b3c9fae6.svg?invert_in_darkmode" align=middle width=39.37145684999999pt height=14.15524440000002pt/> choice for each experiment. In original paper authors use <img src="svgs/590fc0ed1c5b830802797cb17c54a1ee.svg?invert_in_darkmode" align=middle width=70.33014284999999pt height=21.18721440000001pt/>. According to [Tomczak et al., 2018] we initalize set biases mean to zero and standard deviation to one, weights standard deviations are set to <img src="svgs/bd64dd0088109d9c74e867a1de2bbfa7.svg?invert_in_darkmode" align=middle width=33.26498669999999pt height=26.76175259999998pt/> and their means are independent samples from <img src="svgs/e8b5ccce479533dd7ddfaac7376f3526.svg?invert_in_darkmode" align=middle width=119.32198575pt height=47.6716218pt/> for the layer <img src="svgs/7f5282dd648752cd11332df8685daf9c.svg?invert_in_darkmode" align=middle width=57.71787614999998pt height=26.447203200000008pt/>.

### Baseline 2: Hamiltonian Monte Carlo 

<p align="center">
  <img width="500" alt="Hamiltonian Monte Carlo" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/regression/1layer_reg_pool.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Deterministic neural network" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/regression/1layer_reg_det.jpg?raw=true">
</p>

We use **BNN** with 1 layer and <img src="svgs/e91ca1cd7895df20e6f5ab27a52f7207.svg?invert_in_darkmode" align=middle width=78.54935219999999pt height=21.18721440000001pt/>. We train **NUTS** with 5 parallel chains of MCMC and 300 samples from each chain for the distribution estimation and 300 samples for warmup. Result predicition is based on ensemble of these 1500 models of NUTS generated sets of weights. In Pyro we set random seed as **pyro.set_rng_seed(1)** before BNN initialization. We compare our result with training simple deterministic of neural network with the same architecture. For this NN we used **Adam** optimizer with <img src="svgs/07e5d4e4427e3b278e1dab4addc48005.svg?invert_in_darkmode" align=middle width=68.28391679999999pt height=26.76175259999998pt/>, MSE loss and **num epochs = 1000**. We see that deterministic network tends to fit data worse than Bayesian and the Bayesian setting gives smoother results. Results are shown in the 2 figures: the top figure for the NUTS method and the bottom figure for the deterministic neural network. It can be seen, that the uncertainty is higher in the region between two clusters.

### Variational inference: MFVI

#### Custom model 

In this experiment we train 1 layer BNN with the same setup using MFVI approximation family with **ELBO** loss. We estimate ELBO with only one sample (**num particles = 1**), as we discovered that it speeds up the convergence, together with reducing computation time per sample. We trained it using **SVI** class in **Pyro** library with Adam optimizer, <img src="svgs/07e5d4e4427e3b278e1dab4addc48005.svg?invert_in_darkmode" align=middle width=68.28391679999999pt height=26.76175259999998pt/> for **num epochs = 30000** and batch size equal to the whole dataset size. We set random seed as **pyro.set_rng_seed(1)** before training process. Firstly, we show results for different prior choice <img src="svgs/55e3d5b8e0a452ed39463274592ad8be.svg?invert_in_darkmode" align=middle width=138.82339019999998pt height=24.65753399999998pt/> from top to bottom. We can see, that the plain approximator is sensitive to the target’s prior scale.

<p align="center">
  <img width="500" alt="Prior weight standard deviation 0.1" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/1layer_reg_VI_std0.1.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 1" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/1layer_reg_VI_std1.0.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 10" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/1layer_reg_VI_std10.jpg?raw=true">
</p>

We see that optimization process can be unsuccessful for very small prior weight standard deviation: training process is smooth without steps, which indicates only training for uncertainty, but not for the source data in term of mean prediction. The first picture shows that the neural network cannot describe the data well, although the optimization process has converged. We demonstrate loss graphs for the cases <img src="svgs/1f4df7c071fde661775bdc5806e9d30f.svg?invert_in_darkmode" align=middle width=83.1155754pt height=21.18721440000001pt/> and <img src="svgs/e91ca1cd7895df20e6f5ab27a52f7207.svg?invert_in_darkmode" align=middle width=78.54935219999999pt height=21.18721440000001pt/> from top to bottom. 

<p align="center">
  <img width="500" alt="Loss for prior weight standard deviation 0.1" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/losses/1layer_reg_VI_log_loss_bad.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Loss for prior weight standard deviation 10" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/losses/1layer_reg_VI_log_loss.jpg?raw=true">
</p>

#### Model with Local Reparametrization Trick

We emphasize, that Local Reparametrization Trick [Kingma et al., 2015] was used in the original paper. It is believed to simplify the training process due to smaller covariances between the gradients in one batch and, what is more im-
portant, it makes computations more efficient. We implemented this method by scratch. We demonstrate our results for different prior choice <img src="svgs/55e3d5b8e0a452ed39463274592ad8be.svg?invert_in_darkmode" align=middle width=138.82339019999998pt height=24.65753399999998pt/> (from top to bottom) with the same setup as for the custom model.   

<p align="center">
  <img width="500" alt="Prior weight standard deviation 0.1 with trick" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/1layer_reg_VItrick_std0.1.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 1 with trick" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/1layer_reg_VItrick_std1.0.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 10 with trick" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/1layer_reg_VItrick_std10.jpg?raw=true">
</p>

We see that Local Reparametrization Trick poses independence from the prior scale selection. Even though it can make the optimization process more robust, it definitely gives us less control. We compare deeper models: consider the same architecture for the custom model and Local Reparametrization Trick, but with 2 layers. The results are presented in the following figures, top figure corresponds to Local Reparametrization Trick with <img src="svgs/13475692b96681f7f26d35f4d3845e17.svg?invert_in_darkmode" align=middle width=70.33014284999999pt height=21.18721440000001pt/> and the bottom figure corresponds to the custom model with <img src="svgs/e91ca1cd7895df20e6f5ab27a52f7207.svg?invert_in_darkmode" align=middle width=78.54935219999999pt height=21.18721440000001pt/>.

<p align="center">
  <img width="500" alt="Prior weight standard deviation 1 with trick for 2 layers" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/reparametrization_Trick/2layer_reg_VItrick_std1.0.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Prior weight standard deviation 10 for 2 layers" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MFVI/prior_std_tuning/custom_model/2layer_reg_VI_std10.jpg?raw=true">
</p>

Even though it is easier for them to fit the data, there is no significant change in the uncertainty estimation. We emphasize that the usual notion of stacking layers to boost the models complexity doesn’t apply here, so we should keep looking for other approximation techniques.

### Classification 2D dataset and HMC 

Our synthetic 2D classification dataset via regression is the following:
<p align="center"><img src="svgs/07cff3df447f6fb1d0584f2abca8b522.svg?invert_in_darkmode" align=middle width=161.41249079999997pt height=18.312383099999998pt/></p>
<p align="center"><img src="svgs/4a9177a6ed604bad0398a102b8d36a17.svg?invert_in_darkmode" align=middle width=378.32406975pt height=39.452455349999994pt/></p>
<p align="center"><img src="svgs/6a551d95e6f3b8c61eee14cc9116e4fb.svg?invert_in_darkmode" align=middle width=236.40058529999996pt height=18.312383099999998pt/></p>
<p align="center"><img src="svgs/e688a395a1647af8a363e397afb6f5b6.svg?invert_in_darkmode" align=middle width=107.96691344999999pt height=16.438356pt/></p>
We trained 2 layers BNN with the same setup for NUTS as for 1D regression task. We end up with the posterior depicted in the following figures:

<p align="center">
  <img width="500" alt="Classification HMC pool" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/classification/1layer_2d_pool.jpg?raw=true">
</p>
<p align="center">
  <img width="500" alt="Classification HMC contourf" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/HMC/classification/1layer_2d_pool_contourf.jpg?raw=true">
</p>
Unfortunately, we didn't obtain good results for this case. The overall scale of the dataset is the same, as in the regression task, thus it seems possible for the Bayesian model to fit the data given correct hyperparameters, but not for this set of hyperparameters.

### Regression 2D dataset

We present our synthetic 2D regression dataset. Consider two clusters of points <img src="svgs/7392a8cd69b275fa1798ef94c839d2e0.svg?invert_in_darkmode" align=middle width=38.135511149999985pt height=24.65753399999998pt/> with centers in <img src="svgs/5ee11af4458a5d3474b6991426c8f5bc.svg?invert_in_darkmode" align=middle width=95.10524099999999pt height=37.80850590000001pt/> and <img src="svgs/a917556a27ce199605a1518faf7e4623.svg?invert_in_darkmode" align=middle width=69.53437424999998pt height=37.80850590000001pt/> with 100 points in each cluster drawn from normal distributions with standard deviation equal <img src="svgs/22f2e6fc19e491418d1ec4ee1ef94335.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/>. This will be the input variables for our model. The target is simply the evaluation of <img src="svgs/fa95b083b577ff78feb0e4ddfa11a198.svg?invert_in_darkmode" align=middle width=504.74134589999994pt height=24.65753399999998pt/> at these points. Our objective is the uncertainty (standard deviation or variance) predicted by the model for the set <img src="svgs/76638f39555d4e809eca62b73a60e8eb.svg?invert_in_darkmode" align=middle width=111.41555865pt height=24.65753399999998pt/>. 

#### Variance prediction from model and losses from scratch

In these experiments we used our own implementation of BNNs with MFVI and MCDO approximation families based only on **PyTorch** framework. We had to implement BNNs and losses from scratch.

#### Model with unknown conditional variance

This model is analogous to the original with the only distinction, that the conditional variance is unknown and non constant (<img src="svgs/08fd8899ab67b6bd063e2f925f06b8e2.svg?invert_in_darkmode" align=middle width=84.84404114999998pt height=21.18721440000001pt/> in previous examples). Namely, we assume that the conditional distribution of target is given by: <img src="svgs/c52662c4de9c578008b42f684aa45134.svg?invert_in_darkmode" align=middle width=192.73418669999998pt height=32.44583099999998pt/>, i.e. the variance is predicted by the network. In this model the uncertainty is measured by <img src="svgs/32311de6aa16f443b9fd944695f1c5f7.svg?invert_in_darkmode" align=middle width=79.68234284999998pt height=32.44583099999998pt/> and not by <img src="svgs/54bbf88b5ecd15c87c8cf2f4e956a9a9.svg?invert_in_darkmode" align=middle width=78.95446019999999pt height=24.65753399999998pt/> as in the previous cases. In that case BNN has 2 outputs: first is the mean of normal distribution <img src="svgs/2816dde42ea01f596280c5ec74509a2f.svg?invert_in_darkmode" align=middle width=146.00946359999998pt height=34.337843099999986pt/> and the second is <img src="svgs/f3ef48629bc107b21523af678e9fcafc.svg?invert_in_darkmode" align=middle width=80.71627739999998pt height=34.337843099999986pt/>. We will describe how <img src="svgs/f3ef48629bc107b21523af678e9fcafc.svg?invert_in_darkmode" align=middle width=80.71627739999998pt height=34.337843099999986pt/> is connected with <img src="svgs/54bbf88b5ecd15c87c8cf2f4e956a9a9.svg?invert_in_darkmode" align=middle width=78.95446019999999pt height=24.65753399999998pt/> in the section **Custom ELBO loss**.

#### Details of BNNs implementation from scratch

We used the following formulas for forward method in BNNs when we sample weights for current layer <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>:
<p align="center"><img src="svgs/f8dd5d5806721a965bced774d5e3c2a8.svg?invert_in_darkmode" align=middle width=325.38675839999996pt height=17.031940199999998pt/></p>
<p align="center"><img src="svgs/3880b0b322f97734d81ebe609276cad5.svg?invert_in_darkmode" align=middle width=304.45973084999997pt height=17.031940199999998pt/></p>
<p align="center"><img src="svgs/8ce04c4137ce5b195ca0cc2ea8a7d8a4.svg?invert_in_darkmode" align=middle width=68.2362351pt height=17.399144399999997pt/></p>

where <img src="svgs/abfd7ed04f929a076900ab1636d9987b.svg?invert_in_darkmode" align=middle width=30.46914089999999pt height=14.15524440000002pt/> is an matrix with size <img src="svgs/f59e5a2ad72c6e22e9c384cdf31bde0a.svg?invert_in_darkmode" align=middle width=96.44208749999999pt height=22.465723500000017pt/> of learned means for elements in weight matrix <img src="svgs/395e63d84173322e1ee4fbac95c7b8d5.svg?invert_in_darkmode" align=middle width=20.176033349999987pt height=22.465723500000017pt/>; <img src="svgs/cac391e1c61cd4a10ad621f215d2ddf4.svg?invert_in_darkmode" align=middle width=29.063099999999988pt height=14.15524440000002pt/> is an matrix of learned parameters with size <img src="svgs/f59e5a2ad72c6e22e9c384cdf31bde0a.svg?invert_in_darkmode" align=middle width=96.44208749999999pt height=22.465723500000017pt/> that determines standard deviations for weight matrix <img src="svgs/395e63d84173322e1ee4fbac95c7b8d5.svg?invert_in_darkmode" align=middle width=20.176033349999987pt height=22.465723500000017pt/> (matrix <img src="svgs/f606744045f6933ef2b0c88186b9b822.svg?invert_in_darkmode" align=middle width=163.34934495pt height=24.65753399999998pt/> is obtained element-wise from the matrix <img src="svgs/cac391e1c61cd4a10ad621f215d2ddf4.svg?invert_in_darkmode" align=middle width=29.063099999999988pt height=14.15524440000002pt/>); <img src="svgs/afc93854f7d6a423ee5009cc52cfcd0c.svg?invert_in_darkmode" align=middle width=42.26037089999999pt height=21.95701200000001pt/> is an matrix with size <img src="svgs/f59e5a2ad72c6e22e9c384cdf31bde0a.svg?invert_in_darkmode" align=middle width=96.44208749999999pt height=22.465723500000017pt/> such as all elements of <img src="svgs/afc93854f7d6a423ee5009cc52cfcd0c.svg?invert_in_darkmode" align=middle width=42.26037089999999pt height=21.95701200000001pt/> are sampled from <img src="svgs/12283348dea9803642faad8344586e7d.svg?invert_in_darkmode" align=middle width=52.43959049999999pt height=24.65753399999998pt/> independently. The same notation is used for <img src="svgs/847a1cbe0099ba2a1254a723d1266761.svg?invert_in_darkmode" align=middle width=105.59138864999997pt height=21.95701200000001pt/>: <img src="svgs/fffeeb683d4d45ed103a58079fce8570.svg?invert_in_darkmode" align=middle width=24.24079349999999pt height=14.15524440000002pt/> is an vector with size <img src="svgs/cc5188ab1def9c147436f27015f2d3d3.svg?invert_in_darkmode" align=middle width=40.98934619999999pt height=22.465723500000017pt/> of learned means for elements in bias vector <img src="svgs/a4bd6a03f34c02be448007a65da5f451.svg?invert_in_darkmode" align=middle width=11.705695649999988pt height=22.831056599999986pt/>; <img src="svgs/c49660f88b9c460d7d8b9cc855a618c3.svg?invert_in_darkmode" align=middle width=22.83475424999999pt height=14.15524440000002pt/> is an vector of learned parameters with size <img src="svgs/cc5188ab1def9c147436f27015f2d3d3.svg?invert_in_darkmode" align=middle width=40.98934619999999pt height=22.465723500000017pt/> that determines standard deviations for bias vector <img src="svgs/a4bd6a03f34c02be448007a65da5f451.svg?invert_in_darkmode" align=middle width=11.705695649999988pt height=22.831056599999986pt/> (vector <img src="svgs/f606744045f6933ef2b0c88186b9b822.svg?invert_in_darkmode" align=middle width=163.34934495pt height=24.65753399999998pt/> is obtained element-wise from the vector <img src="svgs/c49660f88b9c460d7d8b9cc855a618c3.svg?invert_in_darkmode" align=middle width=22.83475424999999pt height=14.15524440000002pt/>); <img src="svgs/335d51b971bb154f2e2c86675fc225b8.svg?invert_in_darkmode" align=middle width=42.26037089999999pt height=21.95701200000001pt/> is an vector with size <img src="svgs/cc5188ab1def9c147436f27015f2d3d3.svg?invert_in_darkmode" align=middle width=40.98934619999999pt height=22.465723500000017pt/> such as all elements of <img src="svgs/335d51b971bb154f2e2c86675fc225b8.svg?invert_in_darkmode" align=middle width=42.26037089999999pt height=21.95701200000001pt/> are sampled from <img src="svgs/12283348dea9803642faad8344586e7d.svg?invert_in_darkmode" align=middle width=52.43959049999999pt height=24.65753399999998pt/> independently. 

During testing for the model with constant conditional variance we calculate std for BNN output as Monte Carlo estimation using 128 samples of numerical stds for <img src="svgs/00283d79947979941f204a98e74a9c2c.svg?invert_in_darkmode" align=middle width=80.71627739999998pt height=34.337843099999986pt/> according to [Y. K. Foong et al, 2018]. For the model with unknown conditional variance (also during testing) we calculate std for BNN output as Monte Carlo estimation using 128 samples of <img src="svgs/7c9fff384d169ec955a9c68450c809e1.svg?invert_in_darkmode" align=middle width=95.3928426pt height=29.949872699999997pt/>. Also for testing we use BNNs mean output as Monte Carlo estimation using 128 samples of <img src="svgs/00283d79947979941f204a98e74a9c2c.svg?invert_in_darkmode" align=middle width=80.71627739999998pt height=34.337843099999986pt/>. During training these objectives were estimated using 32 Monte Carlo samples according to [Y. K. Foong et al, 2018].

#### ELBO loss from scratch
We implemented an ELBO loss with batches calculation:
<p align="center"><img src="svgs/a87358d893f57ef78746c285150d1d77.svg?invert_in_darkmode" align=middle width=361.31549025pt height=47.60747145pt/></p>
can be rewritten as
<p align="center"><img src="svgs/a4c93fca834685fdd2e088998db6986f.svg?invert_in_darkmode" align=middle width=593.4771380999999pt height=59.1786591pt/></p>

where <img src="svgs/8cfea9645a974b7c12247da8e8a99dd9.svg?invert_in_darkmode" align=middle width=181.15759859999997pt height=47.214766499999996pt/> is a size of train dataset. 

The expectation for log-density part in ELBO loss was estimated using 32 Monte Carlo samples during training according to [Y. K. Foong et al, 2018]. For variance prediction models we used a trick for numerical stability. When calculating log-density Monte-Carlo estimation we implement the following formulas to obtain std prediction <img src="svgs/7c9fff384d169ec955a9c68450c809e1.svg?invert_in_darkmode" align=middle width=95.3928426pt height=29.949872699999997pt/> using the second BNNs output <img src="svgs/f3ef48629bc107b21523af678e9fcafc.svg?invert_in_darkmode" align=middle width=80.71627739999998pt height=34.337843099999986pt/>:
<p align="center"><img src="svgs/9320ee2fe7fd95d0ec18ac969c8d6ba1.svg?invert_in_darkmode" align=middle width=516.49928715pt height=51.5892069pt/></p>

We used <img src="svgs/bb37d900187dc654deaf1940016b7001.svg?invert_in_darkmode" align=middle width=62.848120499999986pt height=26.76175259999998pt/> in all experiments. So, for <img src="svgs/4bfed54bd986e880c5c06c09d307e0b2.svg?invert_in_darkmode" align=middle width=47.75103794999999pt height=21.18721440000001pt/> we rely on <img src="svgs/064f328bc8f1168cc9bf6d69f924ba51.svg?invert_in_darkmode" align=middle width=140.93600399999997pt height=24.65753399999998pt/>. 

#### MFVI using ELBO loss

We trained BNNs on the above regression 2D dataset for 1000 epochs with Adam optimizer with <img src="svgs/bac4bb14ea20e7ba959aed6560d25542.svg?invert_in_darkmode" align=middle width=68.28391679999999pt height=26.76175259999998pt/>.  We present results for training BNNs with 1 and 2 layers for 2 modes: with constant conditional variance <img src="svgs/08fd8899ab67b6bd063e2f925f06b8e2.svg?invert_in_darkmode" align=middle width=84.84404114999998pt height=21.18721440000001pt/> in the model of data <img src="svgs/43dc87a858ec2b3a2ba558c205417909.svg?invert_in_darkmode" align=middle width=228.9386451pt height=26.76175259999998pt/> and with unknown conditional variance in the model of data <img src="svgs/c52662c4de9c578008b42f684aa45134.svg?invert_in_darkmode" align=middle width=192.73418669999998pt height=32.44583099999998pt/>. We used stds estimation that was described in the section **Details of BNNs implementation from scratch**.  Firstly, we show results for 1 layer BNNs, top figure corresponds to the model with constant conditional variance and the bottom figure corresponds to the model with unknown conditional variance.

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
<p align="center"><img src="svgs/2ba4f89716cdfa484652f1bf613dbfdd.svg?invert_in_darkmode" align=middle width=281.26156245pt height=47.93392394999999pt/></p>

where <img src="svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> is the vector of all target values in the training dataset, <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> is the matrix of input variables, <img src="svgs/ed40681200e84cdb8319bd991dec8c22.svg?invert_in_darkmode" align=middle width=48.470454449999984pt height=24.65753399999998pt/> is the vector of expectations of BNN predictions, <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is the number of fully-connected layers, <img src="svgs/7185d0c367d394c42432a1246eceab81.svg?invert_in_darkmode" align=middle width=20.176033349999987pt height=22.465723500000017pt/> and <img src="svgs/d3aa71141bc89a24937c86ec1d350a7c.svg?invert_in_darkmode" align=middle width=11.705695649999988pt height=22.831056599999986pt/> are weights and biases of the <img src="svgs/df443174cee25a90a93c23fb57aced93.svg?invert_in_darkmode" align=middle width=25.68231434999999pt height=24.7161288pt/> layer, for a properly chosen <img src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>. Also it was shown that in order to treat Dropout as Bayesian inference we should choose <img src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/> by the formula:
<p align="center"><img src="svgs/b5caebb4779e8b64ff93de314a9b8cb1.svg?invert_in_darkmode" align=middle width=72.2841009pt height=35.77743345pt/></p>

where <img src="svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270567249999992pt height=14.15524440000002pt/> is the dropout probability, <img src="svgs/80adf96e1bc8b156af4571de2926c45e.svg?invert_in_darkmode" align=middle width=11.780891099999991pt height=26.76175259999998pt/> is the reciprocal of the prior variance for the weigths in the first fully-connected layer, <img src="svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> is the size of the training data and <img src="svgs/0fe1677705e987cac4f589ed600aa6b3.svg?invert_in_darkmode" align=middle width=9.046852649999991pt height=14.15524440000002pt/> is the conditional variance of <img src="svgs/41385193de44a751e47175af1ac510a4.svg?invert_in_darkmode" align=middle width=44.41389644999999pt height=24.65753399999998pt/>. According to [Y. K. Foong et al, 2018] we used <img src="svgs/63faa12e9da736f1c9af3baeafda11f8.svg?invert_in_darkmode" align=middle width=59.41204994999998pt height=21.18721440000001pt/>. Also we used <img src="svgs/bd34a6a146f03d5090f7ef8ef2a7266e.svg?invert_in_darkmode" align=middle width=39.18367364999999pt height=21.18721440000001pt/> and according to notation in [Gal, 2016] <img src="svgs/da3b2a96da9821a88875107013212d7e.svg?invert_in_darkmode" align=middle width=319.0597146pt height=35.75353320000002pt/>. 

#### MCDO results

In this experiment we train BNNs on the regression 2D dataset for 1000 epochs with Adam optimizer with <img src="svgs/07e5d4e4427e3b278e1dab4addc48005.svg?invert_in_darkmode" align=middle width=68.28391679999999pt height=26.76175259999998pt/> using MCDO approximation family with MCDO loss discussed above. We used stds estimation that was described in the section **Details of BNNs implementation from scratch**.  We show results for BNN with 1 and 2 layers from top to bottom.

<p align="center">
  <img width="500" alt="1 HL MCDO BNN" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/std_prediction/1HL_MCDO.jpeg?raw=true">
</p>
<p align="center">
  <img width="500" alt="2 HL MCDO BNN" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/MCDO/std_prediction/2HL_MCDO.jpeg?raw=true">
</p>

We can observe that convexity is preserved in both cases.

#### Baseline for regression 2D dataset: Gaussian Processes

Now we show result for regression task in the section **Regression 2D dataset** using Gaussian Processes from **from sklearn.gaussian_process**. We used **GaussianProcessRegressor** with <img src="svgs/e9457a8f0dfeef1be38656c5a7ef55f0.svg?invert_in_darkmode" align=middle width=200.15684865pt height=22.831056599999986pt/> as a kernel and **n_restarts_optimizer=9** parameter. Parameters for <img src="svgs/1fa1b505cad300dd6666afe2ce01cfab.svg?invert_in_darkmode" align=middle width=57.75893639999999pt height=22.465723500000017pt/> kernel were the following: <img src="svgs/41fe41c99f8fd7df072abd49afcb3435.svg?invert_in_darkmode" align=middle width=296.17588275pt height=26.76175259999998pt/>. Parameters for <img src="svgs/95f390cffd2ae7f8e9d68f48fa99bd38.svg?invert_in_darkmode" align=middle width=122.30672684999999pt height=22.831056599999986pt/> kernel were the following: <img src="svgs/8e1d2fb6f0e3327fa1b2b62935579686.svg?invert_in_darkmode" align=middle width=406.57546545pt height=26.76175259999998pt/>.

<p align="center">
  <img width="500" alt="Gaussian Processes 2D" src="https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/img/GP/Gaussian_Process.png?raw=true">
</p>

We see that the level lines of uncertainty are not convex and display the desirable property: the further away we are from the training points the less confident we are in the prediction. There is no problems with in-between uncertainty or convexity in general.

### Learning BNNs for given mean and variance functions

In this section we are going to the hypothesis that the poor uncertainty prediction is due to improper architecture of neural networks or not. For this task we will try to fit BNNs with certain functions, which play the role of mean and variance functions.

#### Mean/variance functions and loss

We consider functions <img src="svgs/1b30e9275f1aa309d5a0321e4b1cdb63.svg?invert_in_darkmode" align=middle width=113.18931029999999pt height=35.18167619999999pt/> with <img src="svgs/a89dd85d9917c80b990e3c5a180af7c0.svg?invert_in_darkmode" align=middle width=52.346134499999984pt height=21.18721440000001pt/> and <img src="svgs/507bed3b51d1c797ce32ec94b59cbfff.svg?invert_in_darkmode" align=middle width=287.35730759999996pt height=27.94539330000001pt/> on the segment <img src="svgs/43ca5ad9e1f094a31392f860ef481e5c.svg?invert_in_darkmode" align=middle width=45.66218414999998pt height=24.65753399999998pt/>. We will fit different networks using the
following <img src="svgs/7252ad06a4944da2b6628a58281cb887.svg?invert_in_darkmode" align=middle width=11.45742179999999pt height=22.831056599999986pt/> loss function:
<p align="center"><img src="svgs/d1f393a2c3f93df2a77be1644b08adb4.svg?invert_in_darkmode" align=middle width=331.47209864999996pt height=20.611885799999996pt/></p>

#### Train dataset

Our train dataset consists of <img src="svgs/d355bb7facf208832ec1397dfd6aa53d.svg?invert_in_darkmode" align=middle width=53.35601864999999pt height=22.465723500000017pt/> points according to [Y. K. Foong et al, 2018] with values of the functions <img src="svgs/68189a08f1506b5d49556b096a53e948.svg?invert_in_darkmode" align=middle width=69.91449629999998pt height=24.65753399999998pt/> in these points. These points were randomly sampled from the uniform grid of 1000 points on the segment <img src="svgs/43ca5ad9e1f094a31392f860ef481e5c.svg?invert_in_darkmode" align=middle width=45.66218414999998pt height=24.65753399999998pt/>.

#### MFVI training

We train BNNs for 6000 epochs with Adam optimizer, <img src="svgs/07e5d4e4427e3b278e1dab4addc48005.svg?invert_in_darkmode" align=middle width=68.28391679999999pt height=26.76175259999998pt/> using MFVI approximation family with <img src="svgs/7252ad06a4944da2b6628a58281cb887.svg?invert_in_darkmode" align=middle width=11.45742179999999pt height=22.831056599999986pt/> loss from the section **Mean/variance functions and loss**. We present results for training BNNs with 1 and 2 layers for the mode with constant conditional variance. We used stds and means estimations that were described in the section **Details of BNNs implementation from scratch**. We show results for 1 layer and 2 layers BNNs, top 2 figures correspond to BNN with 1 layer and the bottom 2 figures correspond to BNN with 2 layers. <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> axis means indexes of points in grid on the segment <img src="svgs/43ca5ad9e1f094a31392f860ef481e5c.svg?invert_in_darkmode" align=middle width=45.66218414999998pt height=24.65753399999998pt/>.

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

The same conclusion can be done for MCDO approximation family. We also trained MCDO BNNs for 6000 epochs with Adam optimizer, <img src="svgs/07e5d4e4427e3b278e1dab4addc48005.svg?invert_in_darkmode" align=middle width=68.28391679999999pt height=26.76175259999998pt/> and <img src="svgs/7252ad06a4944da2b6628a58281cb887.svg?invert_in_darkmode" align=middle width=11.45742179999999pt height=22.831056599999986pt/> loss from the section **Mean/variance functions and loss**. We used stds and means estimations that were described in the section **Details of BNNs implementation from scratch** for the mode with constant conditional variance. We show results for 1 layer and 2 layers BNNs, top 2 figures correspond to BNN with 1 layer and the bottom 2 figures correspond to BNN with 2 layers. <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> axis means indexes of points in grid on the segment <img src="svgs/43ca5ad9e1f094a31392f860ef481e5c.svg?invert_in_darkmode" align=middle width=45.66218414999998pt height=24.65753399999998pt/>.

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
