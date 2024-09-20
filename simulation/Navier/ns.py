import math
from enum import Enum

import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

class Force(str, Enum):
    li = 'li'
    random = 'random'
    none = 'none'
    kolmogorov = 'kolmogorov'

def get_random_force(b, s, device, cycles, scaling, t, t_scaling, seed):
    ft = torch.linspace(0, 1, s+1).to(device)
    ft = ft[0:-1]   
    X, Y = torch.meshgrid(ft, ft, indexing='ij')
    X = repeat(X, 'x y -> b x y', b=b)
    Y = repeat(Y, 'x y -> b x y', b=b)

    gen = torch.Generator(device)
    gen.manual_seed(seed)

    f = 0
    for p in range(1, cycles + 1):
        k = 2 * math.pi * p

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.sin(k * X + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.cos(k * X + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.sin(k * Y + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.cos(k * Y + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.sin(k * (X + Y) + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.cos(k * (X + Y) + t_scaling * t)

    f = f * scaling

    return f


def generate_force(force, n, s=256, cycles=2, scaling=0.1, seed=44, device='cpu'):
    if force == Force.li:
        # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
        ft = torch.linspace(0, 1, s+1, device=device)
        ft = ft[0:-1]
        X, Y = torch.meshgrid(ft, ft, indexing='ij')
        f = 0.1*(torch.sin(2 * math.pi * (X + Y)) +
                    torch.cos(2 * math.pi * (X + Y)))
        f=f.unsqueeze(0).repeat(n,1,1)
    elif force == Force.kolmogorov:
        ft = torch.linspace(0, 2 * np.pi, s + 1, device=device)
        ft = ft[0:-1]
        X, Y = torch.meshgrid(ft, ft, indexing='ij')
        f = -4 * torch.cos(4 * Y)
        f = f.unsqueeze(0).repeat(n, 1, 1)
    elif force == Force.random:
        f = get_random_force(
            n, s, device, cycles, scaling, 0, 0, seed)
    else:
        f = torch.zeros(n, s, s, device=device)
        # NotImplementedError('Force not implemented')
    return f

def solve_ns_step(w_h, lap, k_x, k_y, delta_t:float, visc, f_h, dealias):
    # Stream function in Fourier space: solve Poisson equation
    psi_h = w_h / lap

    # Velocity field in x-direction = psi_y
    q = psi_h.clone()
    q_real_temp = q.real.clone()
    q = torch.complex(-2 * math.pi * k_y * q.imag, 2 * math.pi * k_y * q_real_temp)
    # q.real = -2 * math.pi * k_y * q.imag
    # q.imag = 2 * math.pi * k_y * q_real_temp
    q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

    # Velocity field in y-direction = -psi_x
    v = psi_h.clone()
    v_real_temp = v.real.clone()
    v = torch.complex(2 * math.pi * k_x * v.imag, -2 * math.pi * k_x * v_real_temp)
    # v.real = 2 * math.pi * k_x * v.imag
    # v.imag = -2 * math.pi * k_x * v_real_temp
    v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

    # Partial x of vorticity
    w_x = w_h.clone()
    w_x_temp = w_x.real.clone()
    w_x = torch.complex(-2 * math.pi * k_x * w_x.imag, 2 * math.pi * k_x * w_x_temp)
    # w_x.real = -2 * math.pi * k_x * w_x.imag
    # w_x.imag = 2 * math.pi * k_x * w_x_temp
    w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

    # Partial y of vorticity
    w_y = w_h.clone()
    w_y_temp = w_y.real.clone()
    w_y = torch.complex(-2 * math.pi * k_y * w_y.imag, 2 * math.pi * k_y * w_y_temp)
    # w_y.real = -2 * math.pi * k_y * w_y.imag
    # w_y.imag = 2 * math.pi * k_y * w_y_temp
    w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

    # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
    F_h = torch.fft.fftn(q * w_x + v * w_y,
                            dim=[1, 2], norm='backward')

    # Dealias
    F_h *= dealias
    
    # Cranck-Nicholson update
    factor = 0.5 * delta_t * visc * lap
    num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
    w_h = num / (1.0 + factor)
    return w_h

solve_jit = torch.jit.script(solve_ns_step)

# def solve_navier_stokes_2d(w0, visc, f, delta_t, steps, record_steps):
#     """Solve Navier-Stokes equations in 2D using Crank-Nicolson method.

#     Parameters
#     ----------
#     w0 : torch.Tensor
#         Initial vorticity field.

#     visc : float
#         Viscosity (1/Re).

#     T : float
#         Final time.

#     delta_t : float
#         Internal time-step for solve (descrease if blow-up).

#     record_steps : int
#         Number of in-time snapshots to record.

#     """

#     # Grid size - must be power of 2
#     N = w0.shape[-1]

#     # Maximum frequency
#     k_max = math.floor(N / 2)

#     # Number of steps to final time
#     # steps = math.ceil(T / delta_t)

#     # Initial vorticity to Fourier space
#     w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

#     f_h = torch.fft.fftn(f, dim=[-2, -1], norm='backward')

#     # Record solution every this number of steps
#     record_time = math.floor(steps / record_steps)

#     # Wavenumbers in y-direction
#     k_y = torch.cat((
#         torch.arange(start=0, end=k_max, step=1, device=w0.device),
#         torch.arange(start=-k_max, end=0, step=1, device=w0.device)),
#         0).repeat(N, 1)
#     # Wavenumbers in x-direction
#     k_x = k_y.transpose(0, 1)

#     # Negative Laplacian in Fourier space
#     lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
#     lap[0, 0] = 1.0

#     # visc = torch.from_numpy(visc).to(w0.device)
#     visc = repeat(visc, 'b -> b m n', m=N, n=N)
#     lap = repeat(lap, 'm n -> b m n', b=w0.shape[0])
#     # visc = visc.unsqueeze(1).unsqueeze(2).repeat(1, N, N)
#     # lap = lap.unsqueeze(0).repeat(w0.shape[0], 1, 1)

#     # Dealiasing mask
#     dealias = torch.unsqueeze(
#         torch.logical_and(
#             torch.abs(k_y) <= (2.0 / 3.0) * k_max,
#             torch.abs(k_x) <= (2.0 / 3.0) * k_max
#         ).float(), 0)
    
#     steps = int(steps)
#     w_list = []
#     for j in tqdm(range(1, steps+1)):
#         w_h = solve_jit(w_h, lap, k_x, k_y, delta_t, visc, f_h, dealias)
#         if j % record_steps == 0:
#             w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
#             w = w.detach().cpu()
#             w_list.append(w)
#     w = torch.stack(w_list, axis=1) # (b, t, n, n)
#     # w = w[:,-1,:,:] # (b, n, n)
#     # w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
#     return w

def solve_navier_stokes_2d(w0, visc, f, delta_t, steps, record_steps):
    """Solve Navier-Stokes equations in 2D using Crank-Nicolson method.

    Parameters
    ----------
    w0 : torch.Tensor
        Initial vorticity field.

    visc : float
        Viscosity (1/Re).

    T : float
        Final time.

    delta_t : float
        Internal time-step for solve (descrease if blow-up).

    record_steps : int
        Number of in-time snapshots to record.

    """

    # Grid size - must be power of 2
    N = w0.shape[-1]

    # Maximum frequency
    k_max = math.floor(N / 2)

    # Number of steps to final time
    # steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

    f_h = torch.fft.fftn(f, dim=[-2, -1], norm='backward')

    # Record solution every this number of steps
    # record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=w0.device),
        torch.arange(start=-k_max, end=0, step=1, device=w0.device)),
        0).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0

    # visc = torch.from_numpy(visc).to(w0.device)
    visc = repeat(visc, 'b -> b m n', m=N, n=N)
    lap = repeat(lap, 'm n -> b m n', b=w0.shape[0])
    # visc = visc.unsqueeze(1).unsqueeze(2).repeat(1, N, N)
    # lap = lap.unsqueeze(0).repeat(w0.shape[0], 1, 1)

    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(
            torch.abs(k_y) <= (2.0 / 3.0) * k_max,
            torch.abs(k_x) <= (2.0 / 3.0) * k_max
        ).float(), 0)
    
    steps = int(steps)
    w_list = []
    for j in range(1, steps+1):
        w_h = solve_jit(w_h, lap, k_x, k_y, delta_t, visc, f_h, dealias)
        if j % record_steps == 0:
            w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
            w_list.append(w)
    w = torch.stack(w_list, axis=1) # (b, t, n, n)
    # w = w[:,-1,:,:] # (b, n, n)
    # w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
    return w