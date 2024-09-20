import os
import sys
import math
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

class ComputePSDiff():
    def __init__(self,N,L,device):
        self.N = N
        self.L = L
        self.device = device

        k = torch.arange(0,N,dtype = torch.float32)
        k[(N+1)//2:] = k[(N+1)//2:]-N
        
        self.k = k
        
    def __call__(self,u,order=1,dim=1,device = None):
        # compute psdiff
        device = self.device if device is None else device
        assert u.shape[dim] == self.N
        
        k = self.k.to(device)
        
        if (order %2 == 1) & (self.N % 2 == 0):
            k[self.N//2] = 0
            
        coeff_shape = [1 if i!=dim else -1 for i in range(len(u.shape))]
        coeff = torch.pow(2j *torch.pi * k/ self.L, order).view(coeff_shape)
        f = torch.fft.fft(u,dim=1)
        df = f * coeff
        du = torch.fft.ifft(df,dim=1)
        return du.real


class PDE(nn.Module):
    """
    Generic PDE template
    """
    def __init__(self):
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def pseudospectral_reconstruction(self):
        """
        A pseudospectral method template
        """
        pass


class KdV(PDE):
    """
    The Korteweg-de Vries equation:
    ut + (0.5*u**2 + uxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'KdV'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device

    # @torch.compile
    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u)
        uxxx = self.psdiff(u, order=3)
        # Compute du/dt.
        dudt = - u*ux - uxxx
        return dudt


class KS(PDE):
    """
    The Kuramoto-Sivashinsky equation:
    ut + (0.5*u**2 + ux + uxxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 40. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 64. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.max_velocity = 0.0

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'KS'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device
        
    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u,)
        uxx = self.psdiff(u, order=2)
        uxxxx = self.psdiff(u, order=4)
        # Compute du/dt.
        dudt = - u*ux - uxx - uxxxx
        return dudt

class Burgers(PDE):
    """
    The Burgers' equation:
    ut + u * ux = nu * uxx
    """
    def __init__(self,
                 nu: float=None,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.device = "cpu"):
        super().__init__()
        # Viscosity coefficient
        self.nu = 0.01 if nu is None else nu
        # Start and end time of the trajectory
        self.tmin = 0.0 if tmin is None else tmin
        self.tmax = 5.0 if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 5 if lmax is None else lmax
        # Number of different sine waves for initial conditions
        self.N = 10
        # Length of the spatial domain
        self.L = 2 * math.pi if L is None else L
        self.grid_size = (100, 256) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = self.grid_size[0] if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # Time and space discretization
        self.dt = self.tmax / (self.nt - 1)
        self.dx = self.L / self.nx
        self.device = device
        # Initialize the pseudo-spectral differentiation operator
        self.psdiff = ComputePSDiff(self.nx, self.L, self.device)
        assert (self.nt >= self.nt_effective)

    def __repr__(self):
        return f'Burgers'

    def set_device(self, device):
        self.device = device
        self.psdiff.device = device

    def pseudospectral_reconstruction_batch(self, t: float, u: torch.Tensor) -> torch.Tensor:
        # u has shape (batch_size, nx)
        # Compute the spatial derivatives using the pseudo-spectral method
        ux = self.psdiff(u, order=1)
        uxx = self.psdiff(u, order=2)
        # Compute du/dt based on Burgers' equation
        dudt = - u * ux + self.nu * uxx
        return dudt


class Heat(PDE):
    def __init__(self,
                 nu: float=None,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Diffusion coefficient
        self.nu = 0.01 if nu is None else nu
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 16. if tmax is None else tmax
        # Sin frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 7 if lmax is None else lmax
        # Number of different waves
        self.N = 20
        # Length of the spatial domain
        self.L = 2 * math.pi if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.alpha = 0.0


        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'Heat'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device
        
    def to_burgers(self,psi: torch.Tensor, device = None):
        # cole-hopf transformation 
        # psi has shape (nt,nx)
        
        psix = self.psdiff(psi,order = 1,device = device)
        return  - (psix / psi) * (2 * self.nu)

    # @torch.compile
    def pseudospectral_reconstruction_batch(self, t: float, u: torch.Tensor) -> torch.Tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        uxx = self.psdiff(u, order=2)
        # Compute du/dt.
        dudt = self.nu * uxx
        return dudt
    
class nKdV(PDE):
    """
    The Korteweg-de Vries equation with nonlinear t:
    ut + exp(t/50)(0.5*u**2 + uxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 2 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'nKdV'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device

    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u)
        uxxx = self.psdiff(u, order=3)
        # Compute du/dt.
        dudt = (- u*ux - uxxx) * torch.exp(t / 50.)
        return dudt
    
class cKdV(PDE):
    """
    The cylindrical Korteweg-de Vries equation:
    ut + (0.5*u**2 + uxx)x + u /(2t+2) = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'cKdV'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device

    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u)
        uxxx = self.psdiff(u, order=3)
        # Compute du/dt.
        dudt = (- u*ux - uxxx - (u/(2*t+2)))
        return dudt


class GrayScott(PDE):
    """
    The Gray-Scott Reaction-Diffusion Model:
    u_t = D_u * Laplacian u - u * v^2 + F * (1 - u)
    v_t = D_v * Laplacian v + u * v^2 - (F + k) * v
    """
    def __init__(self,
                 D_u: float=None,
                 D_v: float=None,
                 F: float=None,
                 k: float=None,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.device = "cpu"):
        super().__init__()
        # Parameters
        self.D_u = 0.16 if D_u is None else D_u
        self.D_v = 0.08 if D_v is None else D_v
        self.F = 0.035 if F is None else F
        self.k = 0.065 if k is None else k
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 8000. if tmax is None else tmax
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        self.N = 10
        # Grid size (nt, nx, ny)
        self.grid_size = (8000, 128, 128) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 8000 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        self.ny = self.grid_size[2]
        # Length of the spatial domain
        self.L = 2.5 if L is None else L
        self.dx = self.L / self.nx
        self.dy = self.L / self.ny
        self.dt = self.tmax / (self.nt - 1)
        self.device = device
        # For 2D FFT, we'll need to adjust ComputePSDiff
        self.psdiff_x = ComputePSDiff(self.nx, self.L, self.device)
        self.psdiff_y = ComputePSDiff(self.ny, self.L, self.device)
        assert (self.nt >= self.nt_effective)

    def __repr__(self):
        return f'GrayScott'

    def set_device(self, device):
        self.device = device
        self.psdiff_x.device = device
        self.psdiff_y.device = device

    def laplacian(self, u):
        u_xx = self.psdiff_x(u, order=2, dim=2)
        u_yy = self.psdiff_y(u, order=2, dim=3)
        return u_xx + u_yy

    def pseudospectral_reconstruction_batch(self, t: float, uv: torch.tensor) -> torch.tensor:
        # uv has shape (batch_size, 2, nx, ny)
        u = uv[:, 0, :, :]
        v = uv[:, 1, :, :]
        # Compute Laplacians
        laplace_u = self.laplacian(u)
        laplace_v = self.laplacian(v)
        # Reaction terms
        uvv = u * v * v
        # Compute RHS
        du_dt = self.D_u * laplace_u - uvv + self.F * (1 - u)
        dv_dt = self.D_v * laplace_v + uvv - (self.F + self.k) * v
        rhs = torch.stack([du_dt, dv_dt], dim=1)
        return rhs