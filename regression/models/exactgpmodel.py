import torch
import gpytorch

class EXACTGPMODEL(gpytorch.models.ExactGP):
    def __init__(self, train_x=None, train_y=None, likelihood=None):
        super(EXACTGPMODEL, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
