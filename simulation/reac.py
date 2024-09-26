import numpy as np
import string, random, os, time
from scipy import interpolate
import subprocess
import torch
import sys
import torch_dct as dct
from tqdm import tqdm

# from hdf5storage import savemat
from hdf5storage import loadmat, savemat

import wandb

from .sim import Sim
import utils
import scipy
from torchdiffeq import odeint

class Reac():
    def __init__(self, tmax=0, fid=32, device='cpu'):
        self.ndim = 1
        self.tmax=0
        self.fid = fid
        self.dim_out = 1
        self.param_dim = self.fid
        self.device=device


    def solve(self, u0):
        # Further updated Parameters
        D = 0.05        # Diffusion coefficient
        epsilon = 10 # Larger parameter for the reaction term
        L = 10.0       # Length of the spatial domain
        N = u0.shape[1]        # Number of spatial points
        T = 5        # Total time for simulation (increased to observe more stable behavior)
        dx = L / (N-1)

        u0 = u0.permute(1,0)

        def reaction_diffusion(t, u):
            # Apply the reaction term
            reaction = (u - u**3) / epsilon
            
            # Apply the diffusion term using finite differences
            u_xx = torch.zeros_like(u)
            u_xx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / dx**2
            u_xx[0] = (u[-1] - 2*u[0] + u[1]) / dx**2  # Periodic boundary condition
            u_xx[-1] = (u[-2] - 2*u[-1] + u[0]) / dx**2  # Periodic boundary condition
            
            return D * u_xx + reaction

        # Time points where solution is computed
        t_eval = torch.linspace(0, T, 2).to(u0.device)

        # Solve the PDE
        # solution = solve_ivp(reaction_diffusion, [0, T], u0, t_eval=t_eval, method='RK45')
        solution = odeint(reaction_diffusion, u0, t_eval)
        return solution[-1].permute(1,0)

    def query_in(self, params):
        # params = xi : bs x s x s where s is the highest fidelity
        bs = params.shape[:-1]
        s = params.shape[-1]
        params = params.view(-1, s)
        fid = self.fid
        X = utils.GRF1D(params)
        # X = X[:,None,None,:] # [bs, 1, 1, s]
        # X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
        # X = X[:,0,0,:] # [bs, fid]
        X = X.view(*bs, fid)

        return X # [bs, fid]

    def query_out(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]

        Y = self.solve(X) # [bs, fid]
        return Y
    

