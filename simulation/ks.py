import numpy as np
import string, random, os, time
from scipy import interpolate
import subprocess
import torch
import sys
import torch_dct as dct
from tqdm import tqdm
from torch.func import jvp

# from hdf5storage import savemat

import utils

class KS():
    def __init__(self, tmax=0, fid=32, device='cpu'):
        self.ndim = 1
        self.fid = fid
        self.dim_out = 1
        self.tmax = tmax
        self.param_dim = self.fid
        self.device=device
        # self.tmax = 10

    def solve(self, u: torch.Tensor):
        # u: [bs, fid]
        assert u.dim() == 2
        device = u.device
        # Initial condition and grid setup
        bs = u.shape[0]
        N = u.shape[1]
        # x = torch.linspace(0, 1, N, device=device)  # Adjust for full 2π range
        # u = torch.rand(bs, N, device=device) * 2 - 1  # Random initial condition for each batch
        v = torch.fft.fft(u, dim=1)  # Apply FFT along each row
        # scalars for ETDRK4
        h = 0.25
        M = 16
        k = torch.cat((torch.arange(0, N//2), torch.tensor([0]), torch.arange(-N//2 + 1, 0))) / 8
        k = k.to(device)
        L = k**2 - k**4
        E = torch.exp(h * L)
        E_2 = torch.exp(h * L / 2)
        r = torch.exp(1j * np.pi * (torch.arange(1, M+1) - 0.5) / M).to(device)
        LR = h * L.repeat(M, 1).T + r.repeat(N, 1)
        Q = h * torch.real((torch.exp(LR/2) - 1) / LR).mean(dim=1)
        f1 = h * torch.real((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3).mean(dim=1)
        f2 = h * torch.real((2 + LR + torch.exp(LR) * (-2 + LR)) / LR**3).mean(dim=1)
        f3 = h * torch.real((-4 - 3 * LR - LR**2 + torch.exp(LR) * (4 - LR)) / LR**3).mean(dim=1)
        # main loop
        # uu = [u.cpu().numpy()]
        # tt = [0]
        tmax = self.tmax
        nmax = round(tmax/h)
        # nplt = int((tmax/100)/h)
        g = -0.5j * k
        for n in range(1, nmax + 1):
            # t = n * h
            Nv = g * torch.fft.fft(torch.real(torch.fft.ifft(v, dim=1)) ** 2, dim=1)
            a = E_2 * v + Q * Nv
            Na = g * torch.fft.fft(torch.real(torch.fft.ifft(a, dim=1)) ** 2, dim=1)
            b = E_2 * v + Q * Na
            Nb = g * torch.fft.fft(torch.real(torch.fft.ifft(b, dim=1)) ** 2, dim=1)
            c = E_2 * a + Q * (2 * Nb - Nv)
            Nc = g * torch.fft.fft(torch.real(torch.fft.ifft(c, dim=1)) ** 2, dim=1)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
            # if n % nplt == 0:
                # u = torch.real(torch.fft.ifft(v, dim=1))
                # uu.append(u.cpu().numpy())
                # tt.append(t)
        u = torch.real(torch.fft.ifft(v, dim=1))
        return u

    def query_in(self, params):
        # params: [*bs, fid]
        bs = params.shape[:-1]
        s = params.shape[-1]
        params = params.view(-1, s)
        fid = self.fid
        X = utils.GRF1D(params) # [bs, s]
        # X = X[:,None,None,:] # [bs, 1, 1, s]
        # X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
        # X = X[:,0,0,:] # [bs, fid]
        X = X.view(*bs, fid)

        return X # [bs, fid]

    def query_out(self, X):
        # X: [*bs, fid]
        bs = X.shape[:-1]
        s = X.shape[-1]
        if X.dim() > 2:
            X = X.view(-1, s)
        Y = self.solve(X)
        Y = Y.view(*bs, s)
        return Y
    