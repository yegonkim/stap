import numpy as np
import torch

import numpy as np
import torch
from torch.func import jvp

from .sim import Sim

class Particles(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.n_particles=500
        self.fid = 2 * self.n_particles
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    
    def query_out_unnorm(self, X):
        # Calculate the average distance between pairs of particles
        # X: [bs, fid]
        # bs = X.shape[0]
        if X.dim() > 2:
            X = X.view(-1, self.fid)
        Y = particles(X)
        Y = Y.view(-1, 1)
        return Y
    


class Hartmann(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 6
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    
    
    def query_out_unnorm(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]
        if X.dim() > 2:
            X = X.view(-1, self.fid)
        Y = hartmann(X)
        Y = Y.view(-1, 1)
        return Y
    


# class Synthetic(Sim):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.ndim = 1
#         self.fid = self.cfg.sim.fid
#         self.cfg = cfg
#         self.ubound = 1.0
#         self.lbound = 0.0
#         self.dim_out = 1
#         self.param_dim = self.fid

#     def query_in_unnorm(self, params):
#         return params.clone()
    
#     def query_out_unnorm(self, X):
#         # X: [bs, fid]
#         # bs = X.shape[0]
#         if X.dim() > 2:
#             X = X.view(-1, self.fid)
#         Y = synthetic1(X)
#         Y = Y.view(-1, 1)
#         return Y

#     def solve(self, x):
#         # Parameters
#         A = 1.0
#         mu = torch.tensor([0.1] * self.fid, dtype=x.dtype, device=x.device)
#         sigma = 0.3
        
#         # Compute the exponent
#         diff = x - mu
#         exponent = -0.5 * torch.sum((diff / sigma) ** 2, dim=1)
        
#         # Compute the function value
#         f_value = A * torch.exp(exponent)
        
#         return f_value.unsqueeze(1)

    

class Synthetic1(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 2
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    
    def query_out_unnorm(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]
        if X.dim() > 2:
            X = X.view(-1, self.fid)
        Y = synthetic(X, self.fid)
        Y = Y.view(-1, 1)
        return Y
    

class Synthetic2(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 6
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    
    def query_out_unnorm(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]
        if X.dim() > 2:
            X = X.view(-1, self.fid)
        Y = synthetic(X, self.fid)
        Y = Y.view(-1, 1)
        return Y


def synthetic(x, fid):
    # Parameters
    A = 1.0
    mu = torch.tensor([0.1] * fid, dtype=x.dtype, device=x.device)
    sigma = 0.5
    
    # Compute the exponent
    diff = x - mu
    exponent = -0.5 * torch.sum((diff / sigma) ** 2, dim=1)
    
    # Compute the function value
    f_value = A * torch.exp(exponent)
    
    return f_value.unsqueeze(1)


class Branin(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 2
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        # params = xi : bs x s x s where s is the highest fidelity
        return params.clone()

    def query_out_unnorm(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]
        return branin_hoo(X)
    
class Borehole(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 8  # Number of fidelity parameters
        self.cfg = cfg
        # self.ubound = torch.tensor([0.15, 50000, 115600, 1110, 820, 116, 1680, 12045])
        self.ubound = 1.0
        # self.lbound = torch.tensor([0.05, 100, 63070, 990, 700, 63.1, 1120, 9855])
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    

    def query_out_unnorm(self, params):
        return borehole_function_batched(params)

    

class Piston(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 7  # Number of fidelity parameters
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def get_params(self, N=None):
        params = torch.rand(N, self.fid) * (self.ubound - self.lbound) + self.lbound
        return params.to(self.cfg.device)

    def query_in_unnorm(self, params):
        return params.clone()
    
    def query_out_unnorm(self, params):
        return piston_simulation_torch(params)


class WingWeight(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 10  # Number of fidelity parameters
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    
    def query_out_unnorm(self, params):
        return wing_weight_torch(params)


class OTLCircuit(Sim):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndim = 1
        self.fid = 5 
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    

    def query_out_unnorm(self, params):
        return otl_circuit_batched(params)



def hartmann(x):
    # Define constants as tensors
    assert x.dim() == 2
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ])
    P = 0.0001 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    # Ensure A, P, and alpha are of the same device and dtype as the input x
    A = A.to(x.device).to(x.dtype)
    P = P.to(x.device).to(x.dtype)
    alpha = alpha.to(x.device).to(x.dtype)

    # Compute the Hartmann 6D function for a batch of inputs
    inner_sum = (x.unsqueeze(1) - P.unsqueeze(0))**2  # Shape [bs, 4, 6]
    inner_sum = torch.sum(A * inner_sum, dim=-1)  # Shape [bs, 4]
    result = -torch.sum(alpha * torch.exp(-inner_sum), dim=-1)  # Shape [bs]

    result = result.unsqueeze(-1)
    return result


def branin_hoo(X):
    x = X[...,0]
    y = X[...,1]
    x = 15 * x - 5
    y = 15 * y
    a = 1.0
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8*np.pi)
    out = a * (y - b*x**2 + c*x - r)**2 + s*(1-t)*torch.cos(x) + s
    return out.unsqueeze(-1)

def borehole_function_batched(parameters):
    """
    Calculate the water flow rate through a borehole using the borehole function for batched inputs, with parameters normalized in the range [0, 1].
    
    Parameters:
    parameters (torch.Tensor): A batch of parameters. Each row should contain normalized values for:
                               [r_w, r, T_u, H_u, H_l, T_l, L, K_w]
                               Shape: [batch_size, 8]
    
    Returns:
    torch.Tensor: Water flow rates in cubic meters per year for each parameter set in the batch.
                  Shape: [batch_size, 1]
    """
    # Map parameters from [0,1] to their respective physical ranges
    r_w = 0.05 + (0.15 - 0.05) * parameters[:, 0]
    r = 100 + (50000 - 100) * parameters[:, 1]
    T_u = 63070 + (115600 - 63070) * parameters[:, 2]
    H_u = 990 + (1110 - 990) * parameters[:, 3]
    H_l = 700 + (820 - 700) * parameters[:, 4]
    T_l = 63.1 + (116 - 63.1) * parameters[:, 5]
    L = 1120 + (1680 - 1120) * parameters[:, 6]
    K_w = 9855 + (12045 - 9855) * parameters[:, 7]
    
    # Compute the natural logarithm term
    ln_term = torch.log(r / r_w)
    
    # Compute the flow rate Q for each set in the batch
    Q = (2 * torch.pi * T_u * (H_u - H_l)) / (ln_term * (1 + (2 * L * T_u) / (ln_term * r_w**2 * K_w) + T_u / T_l))
    
    return Q.unsqueeze(-1)

def piston_simulation_torch(params):
    M = 30 + (60 - 30) * params[:, 0]
    S = 0.005 + (0.020 - 0.005) * params[:, 1]
    V0 = 0.002 + (0.010 - 0.002) * params[:, 2]
    k = 1000 + (5000 - 1000) * params[:, 3]
    P0 = 90000 + (110000 - 90000) * params[:, 4]
    Ta = 290 + (296 - 290) * params[:, 5]
    T0 = 340 + (360 - 340) * params[:, 6]
    
    pi = np.pi
    A = pi * (S / (2 * pi))**2
    V = V0 + (S / (2 * pi)) * torch.sqrt((2 * M) / (k + ((P0 * S) / (2 * pi * Ta))))
    C = 2 * pi * torch.sqrt(M / (k + ((P0 * S) / (2 * pi * Ta)))) + (S / (2 * pi)) * torch.sqrt((2 * M) / (k + ((P0 * S) / (2 * pi * Ta))))
    T = V / A
    out = C + (T / Ta) * (T0 - Ta)
    return out[:,None]

def wing_weight_torch(parameters):
    Sw = 150 + (200 - 150) * parameters[:, 0]
    Wfw = 220 + (300 - 220) * parameters[:, 1]
    A = 6 + (10 - 6) * parameters[:, 2]
    Lambda = -10 + (10 - (-10)) * parameters[:, 3]
    q = 16 + (45 - 16) * parameters[:, 4]
    lam = 0.5 + (1.0 - 0.5) * parameters[:, 5]
    tc = 0.08 + (0.18 - 0.08) * parameters[:, 6]
    Nz = 2.5 + (6.0 - 2.5) * parameters[:, 7]
    Wdg = 1700 + (2500 - 1700) * parameters[:, 8]
    Wp = 0.025 + (0.08 - 0.025) * parameters[:, 9]
    
    Lambda_rad = torch.deg2rad(Lambda)
    cos_Lambda = torch.cos(Lambda_rad)
    term1 = (A / (cos_Lambda**2))**0.6
    term2 = (100 * tc / cos_Lambda)**-0.3
    weight = 0.036 * Sw**0.758 * Wfw**0.0035 * term1 * q**0.006 * lam**0.04 * term2 * Nz**0.49 + Sw * Wp
    return weight.unsqueeze(1)


def otl_circuit_batched(parameters):
    """
    Calculate the OTL Circuit function for batched input using PyTorch tensors, with parameters normalized in the range [0, 1].
    
    Parameters:
    parameters (torch.Tensor): A tensor of shape [batch, 5] containing the normalized resistor values.
                                Each row represents R1, R2, R3, R4, RL for one example.
    
    Returns:
    torch.Tensor: A tensor of shape [batch, 1] with the computed values of the OTL circuit function.
    """
    # Scale the parameters from [0,1] to their respective physical ranges
    R1 = 1000 + (2000 - 1000) * parameters[:, 0]
    R2 = 7000 + (9000 - 7000) * parameters[:, 1]
    R3 = 3000 + (4000 - 3000) * parameters[:, 2]
    R4 = 1000 + (2000 - 1000) * parameters[:, 3]
    RL = 1000 + (2000 - 1000) * parameters[:, 4]
    Rf = 1000.0  # Fixed resistance value in ohms
    
    # Compute voltage V(x) using the given formula
    numerator = Rf * (12 * R4 + 4 * R4 * RL)
    denominator = (R2 + R3 + R4) * (R1 + R4 + RL) + R4 * (R1 + R3 + RL)
    Vx = numerator / denominator
    
    # Compute the OTL circuit function value
    result = (1 - Vx / 10)**2
    
    return result.view(-1, 1)


class Styblinski(Sim):
    def __init__(self, cfg):
        self.ndim = 1
        self.fid = 2
        self.cfg = cfg
        self.ubound = 1.0
        self.lbound = 0.0
        self.dim_out = 1
        self.param_dim = self.fid

    def query_in_unnorm(self, params):
        return params.clone()
    
    def query_out_unnorm(self, X):
        # X: [bs, fid]
        # bs = X.shape[0]
        if X.dim() > 2:
            X = X.view(-1, self.fid)
        Y = styblinski(X)
        Y = Y.view(-1, 1)
        return Y

def styblinski(x):
    x = x * 10 - 5
    out = 0.5 * torch.sum(x**4 - 16 * x**2 + 5 * x, dim=-1, keepdim=True)
    return out

def particles(x):
    # x: [bs, 2*n_particles]
    bs = x.shape[0]
    x_matrix = x.reshape(bs, -1, 2) # [bs, n_particles, 2]
    distances = torch.cdist(x_matrix, x_matrix, p=2) # [bs, n_particles, n_particles]
    avg_distance = torch.mean(distances, dim=[1,2]) # [bs]
    return avg_distance.unsqueeze(1)