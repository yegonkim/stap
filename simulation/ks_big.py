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
from .sim import Sim

class KS_big(Sim):
    def __init__(self, cfg):
        self.ndim = 1
        self.fid = 32
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = -1.0
        self.dim_out = 1
        self.tmax = cfg.sim.tmax
        self.param_dim = self.fid
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
        k = torch.cat((torch.arange(0, N//2), torch.tensor([0]), torch.arange(-N//2 + 1, 0))) / 16
        k = k.to(device)
        L = k**2 - k**4
        E = torch.exp(h * L)
        E_2 = torch.exp(h * L / 2)
        M = 16
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

    # def solve(self, u: torch.Tensor):
    #     # u: [bs, fid]
    #     assert u.dim() == 2
    #     device = u.device
    #     # Initial condition and grid setup
    #     N = u.shape[1]
    #     x = torch.linspace(0, 1, N, device=device)  # Adjust for full 2π range
    #     # u = torch.sin(x *2*np.pi) + 0.5*torch.cos(x*4*np.pi)
    #     # u=torch.rand_like(x)*2-1
    #     v = torch.fft.fft(u)
    #     # scalars for ETDRK4
    #     h = 0.25
    #     k = torch.cat((torch.arange(0, N//2), torch.tensor([0]), torch.arange(-N//2 + 1, 0))) / 16
    #     k = k.to(device)
    #     L = k**2 - k**4
    #     E = torch.exp(h * L)
    #     E_2 = torch.exp(h * L / 2)
    #     M = 16
    #     r = torch.exp(1j * np.pi * (torch.arange(1, M+1) - 0.5) / M).to(device)
    #     LR = h * L.repeat(M, 1).T + r.repeat(N, 1)
    #     Q = h * torch.real((torch.exp(LR/2) - 1) / LR).mean(dim=1)
    #     f1 = h * torch.real((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3).mean(dim=1)
    #     f2 = h * torch.real((2 + LR + torch.exp(LR) * (-2 + LR)) / LR**3).mean(dim=1)
    #     f3 = h * torch.real((-4 - 3 * LR - LR**2 + torch.exp(LR) * (4 - LR)) / LR**3).mean(dim=1)
    #     # main loop
    #     # uu = [u.cpu().numpy()]
    #     tt = [0]
    #     tmax = 150
    #     nmax = round(tmax/h)
    #     nplt = int((tmax/100)/h)
    #     g = -0.5j * k
    #     for n in range(1, nmax + 1):
    #         t = n * h
    #         Nv = g * torch.fft.fft(torch.real(torch.fft.ifft(v)) ** 2)
    #         a = E_2 * v + Q * Nv
    #         Na = g * torch.fft.fft(torch.real(torch.fft.ifft(a)) ** 2)
    #         b = E_2 * v + Q * Na
    #         Nb = g * torch.fft.fft(torch.real(torch.fft.ifft(b)) ** 2)
    #         c = E_2 * a + Q * (2 * Nb - Nv)
    #         Nc = g * torch.fft.fft(torch.real(torch.fft.ifft(c)) ** 2)
    #         v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    #         if n % nplt == 0:
    #             u = torch.real(torch.fft.ifft(v))
    #             # uu.append(u.cpu().numpy())
    #             tt.append(t)

    #     # Convert list of numpy arrays to a single numpy array with np.stack
    #     # uu_array = np.stack(uu)
    #     return u

    # def query_in(self, params):
    #     return params
    
    def query_in_unnorm(self, params):
        return params.clone()

    def query_out_unnorm(self, X):
        # X: [*bs, fid]
        bs = X.shape[:-1]
        s = X.shape[-1]
        if X.dim() > 2:
            X = X.view(-1, s)
        Y = self.solve(X)
        Y = Y.view(*bs, s)
        return Y
    