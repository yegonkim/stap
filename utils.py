import torch
import numpy as np
import random
import math

from omegaconf import OmegaConf
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def torch_delete(tensor, indices):
    """Delete elements from a tensor in indices."""
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def torch_expand(tensor, dim, copies):
    """Expand tensor along a dimension using torch.expand() for memory efficiency."""
    shape = list(tensor.shape)
    shape[dim] = shape[dim] * copies
    return tensor.expand(*shape)

def flatten_configdict(
    cfg: OmegaConf,
    sep: str = ".",
):
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=sep)

    return cfgdict.to_dict(orient="records")[0]

class direct_model(torch.nn.Module):
    def __init__(self, model, unrolling):
        super().__init__()
        self.model = model
        self.unrolling = unrolling
    def forward(self, x):
        for _ in range(self.unrolling):
            x = self.model(x)
        return x
    
class trajectory_model(torch.nn.Module):
    def __init__(self, model, unrolling):
        super().__init__()
        self.model = model
        self.unrolling = unrolling
    def forward(self, x):
        trajectory = []
        for _ in range(self.unrolling):
            x = self.model(x)
            trajectory.append(x)
        return torch.stack(trajectory, dim=1) # [cfg.train.batch_size, unrolling, 1, nx]
    
class split_model(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
    def forward(self, x):
        y = [self.model(x_split) for x_split in x.split(self.batch_size)]
        return torch.cat(y, dim=0)
    
class ensemble_mean_model(torch.nn.Module):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble
    def forward(self, x):
        y = [model(x) for model in self.ensemble]
        return torch.stack(y).mean(dim=0)
    
class normalized_model(torch.nn.Module):
    def __init__(self, model, mean_x, std_x, mean_y, std_y):
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        self.mean_x = mean_x.to(device)
        self.std_x = std_x.to(device)
        self.mean_y = mean_y.to(device)
        self.std_y = std_y.to(device)
    def forward(self, x):
        return self.model((x - self.mean_x) / self.std_x) * self.std_y + self.mean_y
    
class residual_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return x + self.model(x)

class normalized_residual_model(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        self.mean = mean.to(device)
        self.std = std.to(device)
    
    def forward(self, x):
        return x + self.model((x - self.mean) / self.std) * self.std

class zero_mean_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        output = self.model(x) # [bs, c, nx]
        mean = output.mean(dim=tuple(range(2,output.ndim)), keepdim=True)
        return output - mean
    
class zero_mean_constant_energy_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        energy = x.pow(2).sum(dim=tuple(range(2,x.ndim)), keepdim=True)
        output = self.model(x) # [bs, c, nx]
        mean = output.mean(dim=tuple(range(2,output.ndim)), keepdim=True)
        output = output - mean
        output_energy = output.pow(2).sum(dim=tuple(range(2,output.ndim)), keepdim=True)
        output = output * energy.sqrt() / output_energy.sqrt()
        return output

class GaussianRF:
    def __init__(self, n_dims, size, alpha=2, tau=3, sigma=None, device=None):

        self.n_dims = n_dims
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.n_dims))

        k_max = size//2

        if n_dims == 1:
            self.dim = [-1]
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size * \
                math.sqrt(2.0)*sigma * \
                ((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif n_dims == 2:
            self.dim = [-1, -2]
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma * \
                ((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0, 0] = 0.0

        elif n_dims == 3:
            self.dim = [-1, -2, -3]
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)
                                                             * (k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for _ in range(self.n_dims):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, coeff):
        # coeff = torch.randn(N, *self.size, 2, device=self.device)
        # assert self.device == xi.device
        # coeff = xi.view(-1, *self.size, 2)
        coeff = coeff.clone()
        coeff[..., 0] = self.sqrt_eig*coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig*coeff[..., 1]

        u = torch.fft.ifftn(torch.view_as_complex(
            coeff), dim=self.dim, norm='backward').real

        return u
    

def GRF1D(xi, m=0, gamma=2, tau=5, sigma=5**2):
    # xi has shape (bs, s) distributed normally
    xi = xi.clone()
    bs = xi.shape[0:-1]
    s = xi.shape[-1]
    N = s//2 # Example value, adjust as needed

    my_const = 2 * torch.pi

    my_eigs = np.sqrt(2) * (np.abs(sigma) * ((my_const * torch.arange(1, N + 1, device=xi.device))**2 + tau**2) ** (-gamma / 2))
    my_eigs = my_eigs[None]
    xi_alpha = xi[..., :N]
    alpha = my_eigs * xi_alpha
    xi_beta = xi[..., N:]
    beta = my_eigs * xi_beta

    a = alpha / 2
    b = -beta / 2
    c = torch.cat((torch.flip(a,[-1]) - torch.flip(b,[-1]) * 1j, torch.ones(*bs, 1, device=xi.device) * (m + 0j), a + b * 1j), dim=-1)
    field = torch.fft.ifft(torch.fft.ifftshift(c,dim=-1), n=s,dim=-1) * (s)
    field = field.real
    return field
