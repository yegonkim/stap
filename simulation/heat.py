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

class Heat(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 2
        self.fid = 32
        self.cfg = cfg
        self.ubound = 2.0
        self.lbound = -2.0
        self.dim_out = 1
        self.param_dim = self.fid**2 * 2
        self.GRF = utils.GaussianRF(2, self.fid, alpha=2.5, tau=7, sigma=None, device=cfg.device)


    def solve(self, u):
        # Parameters
        alpha = 0.1  # thermal diffusivity
        dx = 1.0  # grid spacing
        dy = 1.0
        dt = 0.1  # time step
        num_steps = 1000  # number of time steps

        # Function to calculate the Laplacian with periodic boundary conditions
        def laplacian_periodic(u):
            lap = torch.zeros_like(u)
            
            lap[:, 1:-1, 1:-1] = (u[:, :-2, 1:-1] + u[:, 2:, 1:-1] + u[:, 1:-1, :-2] + u[:, 1:-1, 2:] - 4 * u[:, 1:-1, 1:-1]) / (dx * dy)
            
            # Edges
            lap[:, 0, 1:-1] = (u[:, -1, 1:-1] + u[:, 1, 1:-1] + u[:, 0, :-2] + u[:, 0, 2:] - 4 * u[:, 0, 1:-1]) / (dx * dy)
            lap[:, -1, 1:-1] = (u[:, -2, 1:-1] + u[:, 0, 1:-1] + u[:, -1, :-2] + u[:, -1, 2:] - 4 * u[:, -1, 1:-1]) / (dx * dy)
            lap[:, 1:-1, 0] = (u[:, 1:-1, -1] + u[:, 1:-1, 1] + u[:, :-2, 0] + u[:, 2:, 0] - 4 * u[:, 1:-1, 0]) / (dx * dy)
            lap[:, 1:-1, -1] = (u[:, 1:-1, -2] + u[:, 1:-1, 0] + u[:, :-2, -1] + u[:, 2:, -1] - 4 * u[:, 1:-1, -1]) / (dx * dy)
            
            # Corners
            lap[:, 0, 0] = (u[:, -1, 0] + u[:, 1, 0] + u[:, 0, -1] + u[:, 0, 1] - 4 * u[:, 0, 0]) / (dx * dy)
            lap[:, -1, 0] = (u[:, -2, 0] + u[:, 0, 0] + u[:, -1, -1] + u[:, -1, 1] - 4 * u[:, -1, 0]) / (dx * dy)
            lap[:, 0, -1] = (u[:, -1, -1] + u[:, 1, -1] + u[:, 0, -2] + u[:, 0, 0] - 4 * u[:, 0, -1]) / (dx * dy)
            lap[:, -1, -1] = (u[:, -2, -1] + u[:, 0, -1] + u[:, -1, -2] + u[:, -1, 0] - 4 * u[:, -1, -1]) / (dx * dy)

            return lap

        # Time-stepping loop
        for step in tqdm(range(num_steps)):
            u = u + alpha * dt * laplacian_periodic(u)
        
        return u
    
    # def solve(self, u):
    #     # Parameters
    #     alpha = 0.01  # thermal diffusivity
    #     dx = 1.0  # grid spacing
    #     dy = 1.0
    #     dt = 0.1  # time step
    #     num_steps = 1000  # number of time steps

    #     # Initialize the temperature grid

    #     # Set initial condition: a hot spot in the center
    #     # u[nx//2, ny//2] = 100.0
    #     # u+= torch.randn_like(u)

    #     # Function to calculate the Laplacian with periodic boundary conditions
    #     def laplacian_periodic(u):
    #         lap = torch.zeros_like(u)
            
    #         lap[1:-1, 1:-1] = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]) / (dx * dy)
            
    #         # Edges
    #         lap[0, 1:-1] = (u[-1, 1:-1] + u[1, 1:-1] + u[0, :-2] + u[0, 2:] - 4 * u[0, 1:-1]) / (dx * dy)
    #         lap[-1, 1:-1] = (u[-2, 1:-1] + u[0, 1:-1] + u[-1, :-2] + u[-1, 2:] - 4 * u[-1, 1:-1]) / (dx * dy)
    #         lap[1:-1, 0] = (u[1:-1, -1] + u[1:-1, 1] + u[:-2, 0] + u[2:, 0] - 4 * u[1:-1, 0]) / (dx * dy)
    #         lap[1:-1, -1] = (u[1:-1, -2] + u[1:-1, 0] + u[:-2, -1] + u[2:, -1] - 4 * u[1:-1, -1]) / (dx * dy)
            
    #         # Corners
    #         lap[0, 0] = (u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0]) / (dx * dy)
    #         lap[-1, 0] = (u[-2, 0] + u[0, 0] + u[-1, -1] + u[-1, 1] - 4 * u[-1, 0]) / (dx * dy)
    #         lap[0, -1] = (u[-1, -1] + u[1, -1] + u[0, -2] + u[0, 0] - 4 * u[0, -1]) / (dx * dy)
    #         lap[-1, -1] = (u[-2, -1] + u[0, -1] + u[-1, -2] + u[-1, 0] - 4 * u[-1, -1]) / (dx * dy)

    #         return lap

    #     # Time-stepping loop
    #     for step in range(num_steps):
    #         u = u + alpha * dt * laplacian_periodic(u)
        
    #     return u

    def query_in_unnorm(self, params):
        # params: [*bs, fid^2 * 2]
        bs = params.shape[:-1]
        params = params.view(*bs, -1)
        coeff = params.view(-1, self.fid, self.fid, 2)
        X = self.GRF.sample(coeff)
        X = X.view(*bs, *X.shape[-2:])

        return X # [bs, fid]

    def query_out_unnorm(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]

        Y = self.solve(X) # [bs, fid]
        return Y
    


# class Burgers(Sim):
#     def __init__(self, cfg):
#         self.M = 1
#         self.ndim = 1
#         self.fidelity_list = [33,129]
#         # self.fidelity_list = [2,4]
#         self.costs = [0.0237, 1-0.0237]
#         self.N_m = len(self.fidelity_list)
#         self.cfg = cfg
#         self.ubound = 6.0
#         self.lbound = 1.0

#     def get_params(self, N=None):
#         params = torch.rand(N,2)*5+1
#         return params.to(self.cfg.device)
        
#     def query_in(self, params, m):
#         # params = [a ; b] : bs x 2
#         if m==-1:
#             m=self.M
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)

#         x = torch.linspace(0,1,self.fidelity_list[m]).to(params.device)
#         X = torch.zeros(params.shape[0],self.fidelity_list[m]).to(params.device)
#         for i in range(params.shape[0]):
#             a = params[i,0]
#             b = params[i,1]
#             X[i,:] = a * torch.exp(-a*x) * torch.sin(2*torch.pi*x) * torch.cos(b*torch.pi*x)
        
#         X = X.view(*bs, X.shape[-1])
#         return X # [bs, n]
        
#     def query_out(self, **kwargs):
#         # params = [a ; b] : bs x 2
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         params = kwargs['params']
#         m = kwargs['m']
#         if m==-1:
#             m=self.M
#         device = params.device

#         params = params.detach().cpu().numpy()
#         if wandb.run is not None:
#             datapath = os.path.join(wandb.run.dir, 'data', '__buff__')
#         else:
#             datapath = 'data/__buff__'
#         if not os.path.exists(datapath):
#             os.makedirs(datapath)

#         query_key = get_random_alphanumeric_string(77)
#         input_path = os.path.join(datapath, query_key + '.mat')
#         query_key = get_random_alphanumeric_string(77)
#         buff_path = os.path.join(datapath, query_key + '.mat')
#         # savemat(input_path, {'params': params, 'fid': float(self.fidelity_list[m])})
#         savemat(input_path, {'X': params, 'fid': float(self.fidelity_list[m])})

#         matlab_cmd = 'addpath(genpath(\'simulation/Burgers\'));'
#         matlab_cmd += f'query_client_burgers(\'{input_path}\', \'{buff_path}\');'
#         matlab_cmd += 'quit force'

#         # print('querying...')
#         command = ["matlab", "-nodesktop", "-r", matlab_cmd]
#         process = subprocess.Popen(command,
#                              stdout=subprocess.PIPE, 
#                              stderr=subprocess.PIPE,
#                              )
#         # process = subprocess.Popen(command,
#         #                      stdout=sys.stdout,
#         #                      stderr=sys.stdout,
#         # )
#         process.wait()
#         # print('done!')
        
#         retrived_data = loadmat(buff_path, squeeze_me=True, struct_as_record=False, mat_dtype=True)['data']

#         Y = retrived_data.Y # [n , t_n, bs]
#         Y = torch.tensor(Y, dtype=torch.float32)
#         if Y.ndim<3:
#             Y = Y.unsqueeze(-1)

#         Y = Y[:,-1,:].permute(1,0) # [bs, n]
#         return Y.to(device)

# class Burgers_grf(Sim):
#     def __init__(self, cfg):
#         self.M = 1
#         self.ndim = 1
#         self.fidelity_list = [32,128]
#         self.costs = [0.0237, 1-0.0237]
#         self.N_m = len(self.fidelity_list)
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0


#     def get_params(self, N=None):
#         params = torch.randn(N, self.fidelity_list[-1])
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params, m):
#         # params = xi : bs x s x s where s is the highest fidelity
#         if m==-1:
#             m=self.M
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fidelity_list[m]
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, **kwargs):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         X = kwargs['X']
#         m = kwargs['m']
#         if m==-1:
#             m=self.M
#         bs = X.shape[0]
#         fid = self.fidelity_list[m]

#         Y = self.solve(X) # [bs, fid]
#         return Y # [bs, fid]

# class Burgers_grf_constrained(Sim):
#     def __init__(self, cfg):
#         self.M = 1
#         self.ndim = 1
#         self.fidelity_list = [32,128]
#         self.costs = [0.0237, 1-0.0237]
#         self.N_m = len(self.fidelity_list)
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0


#     def get_params(self, N=None):
#         params = torch.rand(N, self.fidelity_list[-1]) * (self.ubound - self.lbound) + self.lbound
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params, m):
#         # params = xi : bs x s x s where s is the highest fidelity
#         if m==-1:
#             m=self.M
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fidelity_list[m]
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, **kwargs):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         X = kwargs['X']
#         m = kwargs['m']
#         if m==-1:
#             m=self.M
#         bs = X.shape[0]
#         fid = self.fidelity_list[m]

#         Y = self.solve(X) # [bs, fid]
#         return Y # [bs, fid]

# class Burgers_grf_constrained(Sim):
#     def __init__(self, cfg):
#         self.M = 1
#         self.ndim = 1
#         self.fidelity_list = [32,128]
#         self.costs = [0.0237, 1-0.0237]
#         self.N_m = len(self.fidelity_list)
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0


#     def get_params(self, N=None):
#         params = torch.rand(N, self.fidelity_list[-1]) * (self.ubound - self.lbound) + self.lbound
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params, m):
#         # params = xi : bs x s x s where s is the highest fidelity
#         if m==-1:
#             m=self.M
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fidelity_list[m]
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, **kwargs):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         X = kwargs['X']
#         m = kwargs['m']
#         if m==-1:
#             m=self.M
#         bs = X.shape[0]
#         fid = self.fidelity_list[m]

#         Y = self.solve(X) # [bs, fid]
#         return Y # [bs, fid]
    

# class Burgers_argmax(Sim):
#     def __init__(self, cfg):
#         self.ndim = 1
#         self.fid = 256
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0
#         self.dim_out = 1
        

#     def get_params(self, N=None):
#         params = torch.rand(N, self.fid) * (self.ubound - self.lbound) + self.lbound
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params):
#         # params = xi : bs x s x s where s is the highest fidelity
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fid
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, X):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         bs = X.shape[0]

#         Y = self.solve(X) # [bs, fid]
#         # Y = X
#         pos = torch.argmax(Y, dim=-1)/X.shape[-1] # [bs]
#         return pos.unsqueeze(-1)
    
# class Burgers_argmax_normal(Sim):
#     def __init__(self, cfg):
#         self.ndim = 1
#         self.fid = 256
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0
#         self.dim_out = 1
        

#     def get_params(self, N=None):
#         params = torch.randn(N, self.fid)
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params):
#         # params = xi : bs x s x s where s is the highest fidelity
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fid
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, X):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         bs = X.shape[0]

#         Y = self.solve(X) # [bs, fid]
#         # Y = X
#         pos = torch.argmax(Y, dim=-1)/X.shape[-1] # [bs]
#         return pos.unsqueeze(-1)


# class Burgers_argmax(Sim):
#     def __init__(self, cfg):
#         self.ndim = 1
#         self.fid = 32
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0
#         self.dim_out = 1
        

#     def get_params(self, N=None):
#         params = torch.rand(N, self.fid) * (self.ubound - self.lbound) + self.lbound
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params):
#         # params = xi : bs x s x s where s is the highest fidelity
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fid
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, X):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         bs = X.shape[0]

#         Y = self.solve(X) # [bs, fid]
#         # Y = X
#         pos = torch.argmax(Y, dim=-1)/X.shape[-1] # [bs]
#         return pos.unsqueeze(-1)
    

# class Burgers_argmax_0(Sim):
#     def __init__(self, cfg):
#         self.ndim = 1
#         self.fid = 32
#         self.cfg = cfg
#         self.ubound = 2.0
#         self.lbound = -2.0
#         self.dim_out = 1
        

#     def get_params(self, N=None):
#         params = torch.rand(N, self.fid) * (self.ubound - self.lbound) + self.lbound
#         return params.to(self.cfg.device)

#     def solve(self, u):
#         L = 2*np.pi  # domain length
#         # L= 1
#         bs = u.shape[0]  # batch size
#         N = u.shape[1]  # number of grid points
#         thresh = 32
#         # x = torch.linspace(0, L*(1-1/N), N)  # spatial domain
#         # x = torch.linspace(0, L, N)  # spatial domain

#         nu = 0.1  # viscosity coefficient
#         dt = 0.01  # time step
#         t_end = 1 # end time
#         Nt = int(t_end / dt)  # number of time steps

#         # Step 2: DFT of initial condition
#         u_hat = torch.fft.fft(u, dim=-1)

#         # Wavenumbers: k = 2*pi*n/L for discrete Fourier transform
#         k = torch.fft.fftfreq(N, d=L/N, device=u.device) * 2 * np.pi
#         # print(k)
#         mask = torch.logical_or(k> thresh, k < -thresh)
#         # mask = torch.zeros(N, dtype=torch.bool)
#         k= k.unsqueeze(0)

#         # Time-stepping loop
#         for n in range(Nt):
#             # Transform u_hat back to physical space to compute the nonlinear term
#             # remove one-third of the wavenumbers to avoid aliasing
#             u_hat[:,mask] = 0
#             u = torch.fft.ifft(u_hat, dim=-1)
#             u_hat_nonlinear = torch.fft.fft(-u.real * torch.fft.ifft(1j * k * u_hat, dim=-1).real, dim=-1)
#             u_hat = u_hat + dt * (nu * (1j * k)**2 * u_hat + u_hat_nonlinear)

#         # Step 4: Inverse DFT to get back to physical space
#         u_hat[:,mask] = 0
#         u_final = torch.fft.ifft(u_hat, dim=-1)
#         return u_final.real
    
#     def query_in(self, params):
#         # params = xi : bs x s x s where s is the highest fidelity
#         bs = params.shape[:-1]
#         s = params.shape[-1]
#         params = params.view(-1, s)
#         fid = self.fid
#         X = utils.GRF1D(params) # [bs, s]
#         X = X[:,None,None,:] # [bs, 1, 1, s]
#         X = torch.nn.functional.interpolate(X, size=(1,fid), mode='area') # [bs, 1,1,fid]
#         X = X[:,0,0,:] # [bs, fid]
#         X = X.view(*bs, fid)

#         return X # [bs, fid]

#     def query_out(self, X):
#         # X: [bs, fid]
#         # datapath = os.path.join(wandb.run.dir, 'data')
#         bs = X.shape[0]

#         # Y = self.solve(X) # [bs, fid]
#         Y = X
#         pos = torch.argmax(Y, dim=-1)/X.shape[-1] # [bs]
#         return pos.unsqueeze(-1)
    
