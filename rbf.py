import torch
import torch.nn as nn
import numpy as np


class RBFGrid(nn.Module):

    def __init__(self, ndim, ngrid, lim=None, sigma=0.75, device='cpu'):
        """Projects input on a grid of gaussians with diagonal covariance
        ndim: dimensions of input
        ngrid: width of grid per-dimension
        lim: list of limits per-dimension, e.g. [(-1, 1), (-0.5, 0.2)] for ndim=2
        sigma: standard deviations between each RBF, i.e.:
            Sigma = diag([(lim[i][1] - lim[i][0]) / ngrid / sigma for i in range(ndim)])
        """
        super().__init__()
        if lim is None:
            lim = [(-1, 1)] * ndim
        self.ndim = ndim
        self.ngrid = ngrid
        self.sigma = sigma
        self.d = torch.tensor([(i[1] - i[0])/ngrid*self.sigma for i in lim]).float().to(device)
        # Build a regular grid over each dimension, this will be (ndim, ngrid ** ndim)
        self.grid = torch.tensor(
            np.stack(np.meshgrid(*[np.linspace(*i, ngrid) for i in lim])).reshape((ndim, -1))
        ).float().to(device)

    def forward(self, x):
        # shapes;
        # x: (mbsize, ndim), grid: (ndim, ngrid**2), d: (ndim,)
        dists = (x[:, :, None] - self.grid[None, :, :]) / self.d[None, :, None]
        return 2 * torch.exp(-dists.pow(2).sum(1)) # : (mbsize, ngrid**2)
