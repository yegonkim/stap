import torch
from torch.func import jvp
import numpy as np

import utils

class Sim:
    def __init__(self,cfg=None):
        self.init_norm = False

    def _init_norm(self, normed_dict) :
        self.init_norm = True
        self.X_mean = normed_dict['X_mean']
        self.X_std = normed_dict['X_std']
        self.Y_mean = normed_dict['Y_mean']
        self.Y_std = normed_dict['Y_std']

    def get_ood(self):
        # params: [*bs, fid^2 * 2]
        ood_data = dict()

        for alpha, tau in zip(self.cfg.ood.grf.alpha, self.cfg.ood.grf.tau):
            coeff = torch.randn(self.cfg.ood.grf.num_samples, *((self.fid,)*self.ndim), 2, device=self.cfg.device)
            GRF_ood = utils.GaussianRF(self.ndim, self.fid, alpha=alpha, tau=tau, sigma=None, device=self.cfg.device)
            X = GRF_ood.sample(coeff)
            ood_data[f'grf_{alpha}_{tau}'] = X
        
        for res in self.cfg.ood.perlin.res:
            perlin_ood = utils.Perlin1DSampler(self.fid, res) if self.ndim == 1 else utils.Perlin2DSampler((self.fid, self.fid), (res, res))
            
            X = []
            for _ in range(self.cfg.ood.perlin.num_samples):
                X.append(perlin_ood.sample())
            X = torch.stack(X, dim=0)
            ood_data[f'perlin_{res}'] = X

        return ood_data

    def get_params(self, N=None, mode='randn'):
        dim = self.param_dim
        if mode == 'randn':
            params = torch.randn(N, dim).to(self.cfg.device)
        elif mode == 'rand':
            params = torch.rand(N, dim).to(self.cfg.device) * (self.ubound - self.lbound) + self.lbound
        elif mode == 'linspace':
            params = utils.generate_linspace(N, dim).to(self.cfg.device) * (self.ubound - self.lbound) + self.lbound
        elif mode == 'sobol':
            params = utils.generate_sobol(N, dim).to(self.cfg.device) * (self.ubound - self.lbound) + self.lbound
        return params
        
    def query_in(self, params):
        X = self.query_in_unnorm(params)
        if self.init_norm :
            X = (X - self.X_mean.to(X.device)) / self.X_std.to(X.device)
        return X
    
    def query_in_and_grad(self,p, dp):
        # params: [bs, fid]
        dp = dp / torch.linalg.vector_norm(dp, dim=tuple(range(1, dp.dim())), keepdim=True)
        if self.cfg.sim.gradient_analytic:
            # X, dX = torch.autograd.functional.jvp(self.query_in, p, dp, create_graph=False)
            X, dX = jvp(self.query_in, (p,), (dp,))
        else:
            bs = p.shape[0]
            X = self.query_in(torch.cat([p, p + self.cfg.sim.gradient_step_size * dp], dim=0))
            X, Xv = X[:bs], X[bs:]
            dX = (Xv - X) / self.cfg.sim.gradient_step_size
        # dX = dX / torch.linalg.vector_norm(dX, dim=tuple(range(1, dX.dim())), keepdim=True) * np.sqrt(torch.numel(dX[0]))
        # dX = dX / torch.linalg.vector_norm(dX, dim=tuple(range(1, dX.dim())), keepdim=True)
        dX = dX / torch.linalg.vector_norm(dX, dim=tuple(range(1, dX.dim())), keepdim=True)

        return X, dX

    def query_out(self, X):
        if self.init_norm :
            X = (X * self.X_std.to(X.device)) + self.X_mean.to(X.device)

        Y = self.query_out_unnorm(X).to(self.cfg.device)
        if self.init_norm :
            Y = (Y - self.Y_mean.to(Y.device)) / self.Y_std.to(Y.device)
        
        return Y

    def query_out_and_grad(self, X, dX):
        # X: [bs, fid]
        # V: [bs, fid]

        if self.cfg.sim.gradient_analytic:
            Y, dY = jvp(self.query_out, (X,), (dX,))
        else:
            bs = X.shape[0]
            Y = self.query_out(torch.cat([X, X + self.cfg.sim.gradient_step_size * dX], dim=0))
            Y, Yv = Y[:bs], Y[bs:]
            dY = (Yv - Y) / self.cfg.sim.gradient_step_size

        return Y, dY