import argparse
import os
import sys
import math
import numpy as np
import torch
import h5py
import random
import pickle
import time

from typing import Tuple
from copy import copy
from datetime import datetime
from PDEs import PDE, KdV, nKdV, cKdV, Heat, Burgers
from simulation.ns import NS
from simulation.ks import KS

from tqdm import tqdm

from torchdiffeq import odeint

import hydra
from omegaconf import DictConfig, OmegaConf


def check_files(pde_name: str, modes: dict) -> None:
    """
    Check if data files exist and replace them if wanted.
    Args:
        pde (PDE): pde at hand [KS, KdV, Burgers]
        modes (dict): mode ([train, valid, test]), replace, num_samples, training suffix
    Returns:
            None
    """
    for mode, replace, num_samples in modes:
        save_name = "data/" + "_".join([pde_name, mode])
        save_name = save_name + "_" + str(num_samples)
        if (replace == True):
            if os.path.exists(f'{save_name}.h5'):
                os.remove(f'{save_name}.h5')
                print(f'File {save_name}.h5 is deleted.')
            else:
                print(f'No file {save_name}.h5 exists yet.')
        else:
            print(f'File {save_name}.h5 is kept.')

def check_directory() -> None:
    """
    Check if data and log directories exist, and create otherwise.
    """
    if os.path.exists(f'data'):
        print(f'Data directory exists and will be written to.')
    else:
        os.mkdir(f'data')
        print(f'Data directory created.')

    if not os.path.exists(f'data/log'):
        os.mkdir(f'data/log')

def initial_conditions(A: np.ndarray, phi: np.ndarray, l: np.ndarray, L: float):
    """
    Return initial conditions based on initial parameters.
    Args:
        A (np.ndarray): amplitude of different sine waves
        phi (np.ndarray): phase shift of different sine waves
        l (np.ndarray): frequency of different sine waves
        L (float): length of the spatial domain
    Returns:
        None
    """
    def fnc(x):
        u = np.sum(A * np.sin(2 * np.pi * l * x / L + phi), -1)
        return u
    return fnc

def params(pde: PDE, batch_size: int, device: torch.cuda.device="cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get initial parameters for KdV, KS, and Burgers' equation.
    Args:
        pde (PDE): pde at hand [KS, KdV, Burgers]
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        np.ndarray: amplitude of different sin waves
        np.ndarray: phase shift
        np.ndarray: space dependent frequency
    """
    A = (np.random.rand(1, pde.N) - 0.5) * pde.initial_condition_scale
    phi = 2.0 * np.pi * np.random.rand(1, pde.N)
    l = np.random.randint(pde.lmin, pde.lmax, (1, pde.N))
    return A, phi, l

def inv_cole_hopf(psi0,pde):
    return torch.exp(pde.psdiff((psi0)/(2 * pde.nu),order = -1,dim=1))
    # return torch.exp(pde.psdiff(torch.tensor(psi0,device=pde.psdiff.device)/(2 * pde.nu),order = -1,dim=1))


@torch.no_grad()
def generate_trajectories(pde: PDE,
                          mode: str,
                          num_samples: int,
                          batch_size: int,
                          device: torch.cuda.device="cpu",
                          cfg=None,
                          ) -> None:
    """
    Generate data trajectories for KdV, KS equation on periodic spatial domains.
    Args:
        pde (PDE): pde at hand [KS, KdV, Burgers]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create
        suffix (str): naming suffix for special trajectories
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        None
    """

    num_batches = int(np.ceil(num_samples / batch_size))

    pde_string = str(pde)
    print(f'Equation: {pde_string}')
    print(f'Mode: {mode}')
    print(f'Number of samples: {num_samples}')

    sys.stdout.flush()

    save_name = "data/" + "_".join([pde_string, mode])
    save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    dataset = h5f.create_group(mode)

    h5f_u = {}

    # Tolerance of the solver
    atol = cfg.generate_data.atol
    rtol = cfg.generate_data.rtol
    
    
    nt = pde.grid_size[0]
    nx = pde.grid_size[1]
    
    # The field u, the coordinations (xcoord, tcoord) and dx, dt are saved
    # Only nt_effective time steps of each trajectories are saved
    h5f_u = dataset.create_dataset(f'pde', (num_samples, pde.nt_effective, 1, nx), dtype=float)
        
    for batch_idx in range(num_batches):
        
        time_start = time.time()

        n_data = min((batch_idx+1) * batch_size,num_samples) - batch_idx * batch_size
        if n_data == 0:
            continue
        u0_list = []
        for trial in range(5):
            try:
                for j in range(n_data):
                    
                    T = pde.tmax
                    L = pde.L

                    t = np.linspace(pde.tmin, T, nt)
                    x = np.linspace(0, (1 - 1.0 / nx) * L, nx)

                    # Parameters for initial conditions
                    A, omega, l = params(pde, batch_size, device=device)

                    # Initial condition of the equation at end
                    u0 = initial_conditions(A, omega, l, L)(x[:, None])

                    u0_list.append(u0)
                    
            
                spectral_method = pde.pseudospectral_reconstruction_batch
                        
                u0 = torch.tensor(np.stack(u0_list,axis=0)).to(device)
                if pde_string == 'Burgers':
                    u0 = inv_cole_hopf(u0, pde)
                t = torch.tensor(t).to(device)

                solved_trajectory = odeint(func=spectral_method,
                                            t=t,
                                            y0=u0,
                                            method=cfg.generate_data.solver,
                                            atol=atol,
                                            rtol=rtol)
                break
            except AssertionError:
                print(f'An error occured - possibly an underflow. re-running {trial}/5')

        time_end = time.time()

        sol = solved_trajectory.permute(1,0,2)[:,-pde.nt_effective:]
        # if pde_string == 'Burgers':
        #     sol = pde.to_burgers(sol.reshape(-1,nx)).reshape(*sol.shape)
        #     sol = torch.tensor(cole_hopf(sol.cpu().numpy()))
            
        sol = sol.cpu()
        h5f_u[batch_idx * batch_size:batch_idx * batch_size+n_data, :, 0, :] = sol
        
        print("Solved indices: {:d} : {:d}".format(batch_idx * batch_size, (batch_idx + 1) * batch_size - 1))
        print("Solved batches: {:d} of {:d}".format(batch_idx + 1, num_batches))
        print("Time elapsed: {:.2f} seconds".format(time_end - time_start))
        sys.stdout.flush()

    print()
    print("Data saved")
    print()
    print()
    h5f.close()

def generate_data(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  device: torch.cuda.device="cpu",
                  nu: float=0.01,
                  cfg=None) -> None:
    if experiment in ['KdV', 'Heat', 'nKdV', 'cKdV']:
        _generate_data_ps(experiment, starting_time, end_time, L, nx, nt, nt_effective, num_samples_train, num_samples_valid, num_samples_test, batch_size, device, nu, cfg)
    elif experiment in ['NS', 'KS']:
        _generate_data_sim(experiment, starting_time, end_time, L, nx, nt, nt_effective, num_samples_train, num_samples_valid, num_samples_test, batch_size, device, nu, cfg)
    elif experiment in ['Burgers', 'CNS']:
        _generate_data_jax(experiment, starting_time, end_time, L, nx, nt, nt_effective, num_samples_train, num_samples_valid, num_samples_test, batch_size, device, nu, cfg)
    else:
        raise Exception("Wrong experiment")


def _generate_data_jax(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  device: torch.cuda.device="cpu",
                  nu: float=0.01,
                  cfg=None) -> None:
    print(f'Generating data')
    from al4pde.tasks.ic_gen.ic_gen_burgers import ICGenBurgers
    from al4pde.tasks.sim.burgers import BurgersSim
    from al4pde.tasks.ic_gen.ic_gen_2d_ns_rand import ICGenNSRand
    from al4pde.tasks.param_gen import PDEParamGenerator
    from al4pde.tasks.sim.cfd import CFDSim


    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", num_samples_train > 0, num_samples_train),
             ("valid", num_samples_valid > 0, num_samples_valid),
             ("test", num_samples_test > 0, num_samples_test)}
    check_files(experiment, files)

    if experiment == 'Burgers':
        assert nt == cfg.generate_data.nt
        dt = cfg.generate_data.end_time / (cfg.generate_data.nt-1)
        nu = cfg.generate_data.nu
        nx = cfg.generate_data.nx
        L = cfg.generate_data.L
        sim = BurgersSim(
            ini_time=0.,
            dt=dt,        
            CFL=0.25,       
            show_steps=100,
            if_norm=False, 
            if_second_order=1.,
        )
        ic_gen = ICGenBurgers(
            k_tot = 4,
            num_choice_k = 2,
            xL = 0.,
            xR = L,
            nx = nx,
        )
        pde_params = torch.Tensor([[nu]]).repeat(batch_size, 1)
        grid = ic_gen.get_grid(1)[0]

    elif experiment == 'CNS':
        assert nt == cfg.generate_data.nt
        L = cfg.generate_data.L
        nx = cfg.generate_data.nx
        T = cfg.generate_data.end_time
        nt = cfg.generate_data.nt 
        dt = cfg.generate_data.end_time / (cfg.generate_data.nt-1)
        
        eta = cfg.generate_data.eta
        zeta = cfg.generate_data.zeta
        ic_gen = ICGenNSRand(
            k_tot = 4,
            xL = 0.,
            xR = L,
            yL = 0.0,
            yR = L,
            zL = 0.0,
            zR = 1.0,
            nx = nx,
            ny = nx,
            nz = 1,
            mach_min = 0.1,
            mach_max = 1.0,
            gamma = 1.6666666666666667,
            d0Min = 0.1,
            d0Max = 10.0,
            T0Min = 0.1,
            T0Max = 10.0,
            init_field_type = 'rand',
            constrain_max = True,
            delDMin = 0.013,
            delDMax = 0.26,
            delPMin = 0.04,
            delPMax = 0.8,
        )
        sim = CFDSim(
            pde_name = 'CFD_2D_Rand_S',
            same_eta_zeta = False,
            spatial_dim = 2,
            ini_time = 0.0,
            fin_time = T,
            dt = dt,
            CFL = 0.3,
            show_steps = 100,
            if_second_order = 1.0,
            bc = 'periodic',
            gamma = 1.6666666666666667,
            p_floor = 0.0001,
        )
        pde_params = torch.tensor([[eta,zeta]]).repeat(batch_size,1)

        grid = ic_gen.get_grid(1)[0]
    else:
        raise Exception("Wrong experiment")
    
    if experiment == 'Burgers':
        for mode, _, num_samples in files:
            if num_samples > 0:
                
                num_batches = int(np.ceil(num_samples / batch_size))

                pde_string = experiment
                print(f'Equation: {pde_string}')
                print(f'Mode: {mode}')
                print(f'Number of samples: {num_samples}')

                sys.stdout.flush()

                save_name = "data/" + "_".join([pde_string, mode])
                save_name = save_name + "_" + str(num_samples)
                h5f = h5py.File("".join([save_name, '.h5']), 'a')
                dataset = h5f.create_group(mode)
                h5f_u = {}
                h5f_u = dataset.create_dataset(f'pde', (num_samples, nt_effective, 1, nx), dtype=float)

                # for batch_idx in tqdm(range(num_batches)):
                for batch_idx in range(num_batches):
                    n_data = min((batch_idx+1) * batch_size,num_samples) - batch_idx * batch_size
                    if n_data == 0:
                        continue
                    ic_params = ic_gen.initialize_ic_params(batch_size)
                    ic = ic_gen.generate_initial_conditions(ic_params, pde_params)
                    try:
                        traj, _= sim(ic, pde_params,grid, nt=nt)

                    except AssertionError:
                        raise Exception('An error occured while generating data.')
                    traj = traj[:, -nt_effective:,None,:,0] # (n_data, nt_effective, 1, nx)
                    traj = traj.cpu()
                    h5f_u[batch_idx * batch_size:batch_idx * batch_size+n_data] = traj[:n_data]
            
                print()
                print("Data saved")
                print()
                print()
                h5f.close()
    elif experiment == 'CNS':
        multiplier = 5
        for mode, _, num_samples in files:
            if num_samples > 0:
                
                num_batches = int(np.ceil(num_samples / batch_size))

                pde_string = experiment
                print(f'Equation: {pde_string}')
                print(f'Mode: {mode}')
                print(f'Number of samples: {num_samples}')

                sys.stdout.flush()

                save_name = "data/" + "_".join([pde_string, mode])
                save_name = save_name + "_" + str(num_samples)
                h5f = h5py.File("".join([save_name, '.h5']), 'a')
                dataset = h5f.create_group(mode)
                h5f_u = {}
                h5f_u = dataset.create_dataset(f'pde', (num_samples, nt_effective*multiplier, 4, nx, nx), dtype=float)

                # for batch_idx in tqdm(range(num_batches)):
                for batch_idx in range(num_batches):
                    n_data = min((batch_idx+1) * batch_size,num_samples) - batch_idx * batch_size
                    if n_data == 0:
                        continue
                    ic_params = ic_gen.initialize_ic_params(batch_size)
                    ic = ic_gen.generate_initial_conditions(ic_params, pde_params)
                    try:
                        traj, _, _= sim.n_step_sim(ic, pde_params,grid, 0,n_steps=nt) # (n_data, nt, nx, nx, 4)

                    except AssertionError:
                        raise Exception('An error occured while generating data.')
                    traj = traj[:, -nt_effective:].permute(0,1,4,2,3) # (n_data, nt_effective, 4, nx, nx)
                    traj = traj.repeat_interleave(multiplier, dim=1)  # Repeat each timestep 5 times along time dimension (n_data, nt_effective*5, 4, nx, nx)
                    traj = traj.cpu()
                    h5f_u[batch_idx * batch_size:batch_idx * batch_size+n_data] = traj[:n_data]
            
                print()
                print("Data saved")
                print()
                print()
                h5f.close()

        

def _generate_data_sim(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  device: torch.cuda.device="cpu",
                  nu: float=0.01,
                  cfg=None) -> None:
    print(f'Generating data')


    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", num_samples_train > 0, num_samples_train),
             ("valid", num_samples_valid > 0, num_samples_valid),
             ("test", num_samples_test > 0, num_samples_test)}
    check_files(experiment, files)

    if experiment == 'NS':
        assert nt == cfg.generate_data.nt
        dt = cfg.generate_data.end_time / (nt - 1)
        dt_sim = cfg.generate_data.dt
        vis = cfg.generate_data.vis
        fid = cfg.generate_data.nx
        force = cfg.generate_data.force
        sim = NS(tmax=dt, dt=dt_sim, vis=vis, fid=fid, force=force, device=device)
    elif experiment == 'KS':
        assert nt == cfg.generate_data.nt
        dt = cfg.generate_data.end_time / (nt - 1)
        fid = cfg.generate_data.nx
        sim = KS(tmax=dt, fid=fid, device=device)
    else:
        raise Exception("Wrong experiment")
    
    if experiment == 'NS':
        for mode, _, num_samples in files:
            if num_samples > 0:
                
                num_batches = int(np.ceil(num_samples / batch_size))

                pde_string = experiment
                print(f'Equation: {pde_string}')
                print(f'Mode: {mode}')
                print(f'Number of samples: {num_samples}')

                sys.stdout.flush()

                save_name = "data/" + "_".join([pde_string, mode])
                save_name = save_name + "_" + str(num_samples)
                h5f = h5py.File("".join([save_name, '.h5']), 'a')
                dataset = h5f.create_group(mode)
                h5f_u = {}
                h5f_u = dataset.create_dataset(f'pde', (num_samples, nt_effective, 1, nx, nx), dtype=float)

                # for batch_idx in tqdm(range(num_batches)):
                for batch_idx in range(num_batches):
                    n_data = min((batch_idx+1) * batch_size,num_samples) - batch_idx * batch_size
                    if n_data == 0:
                        continue
                    params = torch.randn(n_data, nx**2 * 2).to(device)
                    u0 = sim.query_in(params).unsqueeze(1) # (n_data, 1, nx, nx)
                    try:
                        traj = [u0]
                        for i in range(nt - 1):
                            traj.append(sim.query_out(traj[-1].squeeze(1)).unsqueeze(1)) # (n_data, 1, nx, nx)
                        traj = torch.stack(traj, dim=1) # (n_data, nt, 1, nx, nx)
                    except AssertionError:
                        raise Exception('An error occured - possibly an underflow.')
                    traj = traj[:, -nt_effective:] # (n_data, nt_effective, 1, nx, nx)
                    traj = traj.cpu()
                    h5f_u[batch_idx * batch_size:batch_idx * batch_size+n_data] = traj
            
                print()
                print("Data saved")
                print()
                print()
                h5f.close()

    if experiment == 'KS':
        for mode, _, num_samples in files:
            if num_samples > 0:
                
                num_batches = int(np.ceil(num_samples / batch_size))

                pde_string = experiment
                print(f'Equation: {pde_string}')
                print(f'Mode: {mode}')
                print(f'Number of samples: {num_samples}')

                sys.stdout.flush()

                save_name = "data/" + "_".join([pde_string, mode])
                save_name = save_name + "_" + str(num_samples)
                h5f = h5py.File("".join([save_name, '.h5']), 'a')
                dataset = h5f.create_group(mode)
                h5f_u = {}
                h5f_u = dataset.create_dataset(f'pde', (num_samples, nt_effective, 1, nx), dtype=float)

                for batch_idx in range(num_batches):
                    n_data = min((batch_idx+1) * batch_size,num_samples) - batch_idx * batch_size
                    if n_data == 0:
                        continue
                    params = torch.randn(n_data, nx).to(device)
                    u0 = sim.query_in(params).unsqueeze(1) # (n_data, 1, nx)
                    try:
                        traj = [u0]
                        for i in range(nt - 1):
                            traj.append(sim.query_out(traj[-1].squeeze(1)).unsqueeze(1)) # (n_data, 1, nx)
                        traj = torch.stack(traj, dim=1) # (n_data, nt, 1, nx)
                    except AssertionError:
                        raise Exception('An error occured - possibly an underflow.')
                    traj = traj[:, -nt_effective:] # (n_data, nt_effective, 1, nx)
                    traj = traj.cpu()
                    h5f_u[batch_idx * batch_size:batch_idx * batch_size+n_data] = traj
            
                print()
                print("Data saved")
                print()
                print()
                h5f.close()

def _generate_data_ps(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  device: torch.cuda.device="cpu",
                  nu: float=0.01,
                  cfg=None) -> None:
    """
    Generate data for KdV, KS equation on periodic spatial domains.
    Args:
        experiment (str): pde at hand [KS, KdV, Heat]
        starting_time (float): starting time of PDE solving
        end_time (float): end time of PDE solving
        L (float): length of the spatial domain
        nx (int): spatial resolution
        nt (int): temporal resolution
        num_samples_train (int): number of trajectories created for training
        num_samples_valid (int): number of trajectories created for validation
        num_samples_test (int): number of trajectories created for testing
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        None
    """
    print(f'Generating data')
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    # if args.log:
    #     logfile = f'data/log/{experiment}_time{timestring}.csv'
    #     print(f'Writing to log file {logfile}')
    #     sys.stdout = open(logfile, 'w')

    # Create instances of PDE
    if experiment == 'KdV':
        pde = KdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'KS':
        pde = KS(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 L=L,
                 device=device)
    elif experiment == 'Heat':
        pde = Heat(tmin=starting_time,
                    tmax=end_time,
                    grid_size=(nt, nx),
                    nt_effective=nt_effective,
                    nu=nu,
                    L=L,
                    device=device)
    elif experiment == 'Burgers':
        # Heat equation is generated; afterwards trajectories are transformed via Cole-Hopf transformation.
        # L is not set for Burgers equation, since it is very sensitive. Default value is 2*math.pi.
        pde = Burgers(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 nu=nu,
                 device=device)
    elif experiment == 'nKdV':
        pde = nKdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'cKdV':
        pde = cKdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    else:
        raise Exception("Wrong experiment")

    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", num_samples_train > 0, num_samples_train),
             ("valid", num_samples_valid > 0, num_samples_valid),
             ("test", num_samples_test > 0, num_samples_test)}
    check_files(str(pde), files)
    
    # save pde object
    # with open(os.path.join('data',f'{str(pde)}.pkl'),'wb') as f:
    #     pickle.dump(pde,f)

    # Obtain trajectories for different modes
    for mode, _, num_samples in files:
        if num_samples > 0:
            generate_trajectories(pde=pde,
                                mode=mode,
                                num_samples=num_samples,
                                batch_size=batch_size,
                                device=device,
                                cfg=cfg)


@hydra.main(version_base=None, config_path="cfg_flexible", config_name="config.yaml")
def main(cfg: OmegaConf):
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".90" # proportion of memory preallocated for jax
    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    check_directory()

    start_time = time.time()
    generate_data(experiment=cfg.equation,
                    starting_time=cfg.generate_data.starting_time,
                    end_time=cfg.generate_data.end_time,
                    L=cfg.generate_data.L,
                    nt=cfg.generate_data.nt,
                    nt_effective=cfg.generate_data.nt_effective,
                    nx=cfg.generate_data.nx,
                    num_samples_train=cfg.datasize,
                    num_samples_valid=0,
                    num_samples_test=cfg.testsize,
                    batch_size=cfg.generate_data.batch_size,
                    device=cfg.device,
                    nu=cfg.generate_data.nu,
                    cfg=cfg,)
    end_time = time.time()
    print(f'Time elapsed: {end_time - start_time:.2f} seconds')



if __name__ == "__main__":
    main()


def _get_pde_object(cfg):
    experiment = cfg.equation
    starting_time = 0.0
    end_time = cfg.generate_data.end_time
    nt, nx = cfg.generate_data.nt, cfg.generate_data.nx
    L = cfg.generate_data.L
    device = cfg.device
    nt_effective = nt

    # Create instances of PDE
    if experiment == 'KdV':
        pde = KdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'KS':
        pde = KS(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 L=L,
                 device=device)
    elif experiment=='Heat':
        pde = Heat(tmin=starting_time,
                   tmax=end_time,
                   grid_size=(nt, nx),
                   nt_effective=nt_effective,
                   L=L,
                   device=device)
    elif experiment == 'Burgers':
        # Heat equation is generated; afterwards trajectories are transformed via Cole-Hopf transformation.
        # L is not set for Burgers equation, since it is very sensitive. Default value is 2*math.pi.
        pde = Burgers(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 nu=cfg.generate_data.nu,
                 device=device)
    elif experiment == 'nKdV':
        pde = nKdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'cKdV':
        pde = cKdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    else:
        raise Exception("Wrong experiment")

    return pde



@torch.no_grad()
def evolve(u0, cfg, t0=0, timesteps=1):
    if cfg.equation in ["KdV", "Heat", "nKdV", "cKdV"]:
        return _evolve_ps(u0, cfg, t0, timesteps)
    elif cfg.equation in ["NS", "KS"]:
        return _evolve_sim(u0, cfg, t0, timesteps)
    elif cfg.equation == "Burgers":
        evolve_burgers = EvolveJax(cfg)
        timestep_length = 130//(cfg.nt-1)
        return evolve_burgers(u0, t0, timesteps*timestep_length)[:,timestep_length::timestep_length]
    elif cfg.equation == "CNS":
        evolve_cns = EvolveJaxCNS(cfg)
        return evolve_cns(u0, t0, timesteps)[:,1:]
        # timestep_length = 130//(cfg.nt-1)
        # return evolve_cns(u0, t0, timesteps*timestep_length)[:,timestep_length::timestep_length]
    else:
        raise Exception("Wrong experiment")

class EvolveJax():
    def __init__(self, cfg):
        from al4pde.tasks.ic_gen.ic_gen_burgers import ICGenBurgers
        from al4pde.tasks.sim.burgers import BurgersSim
        self.cfg = cfg
        nt = cfg.generate_data.nt
        dt = cfg.generate_data.end_time / (cfg.generate_data.nt-1)
        nu = cfg.generate_data.nu
        nx = cfg.generate_data.nx
        L = cfg.generate_data.L
        self.nu = nu
        self.sim = BurgersSim(
            ini_time=0.,
            dt=dt,
            CFL=0.25,
            show_steps=100,
            if_norm=False,
            if_second_order=1.,
        )
        ic_gen = ICGenBurgers(
            k_tot = 4, # not used, no need to change
            num_choice_k = 2, # not used, no need to change
            xL = 0.,
            xR = L,
            nx = nx,
        )
        self.grid = ic_gen.get_grid(1)[0]
    def __call__(self, u0, t0=0, timesteps=1):
        # u0 has shape (n_data, 1, nx)
        n_data, _, nx = u0.shape
        timesteps = timesteps + 1
        onedata = False
        if n_data == 1:
            u0 = u0.repeat(2,1,1)
            n_data = 2
            onedata = True
        u0 = u0.reshape(n_data,nx,1,1)
        pde_params = torch.Tensor([[self.nu]]).repeat(n_data, 1)
        traj, _ = self.sim(u0,pde_params,self.grid, nt=timesteps)
        traj = traj.reshape(n_data,timesteps,1,nx)
        if onedata:
            traj = traj[:1]
        return traj


class EvolveJaxCNS():
    def __init__(self, cfg):
        from al4pde.tasks.ic_gen.ic_gen_2d_ns_rand import ICGenNSRand
        from al4pde.tasks.sim.cfd import CFDSim
        self.cfg = cfg
        
        L = cfg.generate_data.L
        nx = cfg.generate_data.nx
        # T = cfg.generate_data.end_time
        # nt = cfg.generate_data.nt 
        dt = cfg.generate_data.end_time / (cfg.generate_data.nt-1) * (26 / (cfg.nt-1))
        
        eta = cfg.generate_data.eta
        zeta = cfg.generate_data.zeta
        
        self.eta = eta
        self.zeta = zeta
        ic_gen = ICGenNSRand(
            k_tot = 4,
            xL = 0.,
            xR = L,
            yL = 0.0,
            yR = L,
            zL = 0.0,
            zR = 1.0,
            nx = nx,
            ny = nx,
            nz = 1,
            mach_min = 0.1,
            mach_max = 1.0,
            gamma = 1.6666666666666667,
            d0Min = 0.1,
            d0Max = 10.0,
            T0Min = 0.1,
            T0Max = 10.0,
            init_field_type = 'rand',
            constrain_max = True,
            delDMin = 0.013,
            delDMax = 0.26,
            delPMin = 0.04,
            delPMax = 0.8,
        )
        self.sim = CFDSim(
            pde_name = 'CFD_2D_Rand_S',
            same_eta_zeta = False,
            spatial_dim = 2,
            ini_time = 0.0,
            fin_time = dt,
            dt = dt,
            CFL = 0.3,
            show_steps = 100,
            if_second_order = 1.0,
            bc = 'periodic',
            gamma = 1.6666666666666667,
            p_floor = 0.0001,
        )
        # pde_params = torch.tensor([[eta,zeta]]).repeat(batch_size,1)

        self.grid = ic_gen.get_grid(1)[0]
        
    def __call__(self, u0, t0=0, timesteps=1):
        # u0 has shape (n_data, 4, nx,nx)
        n_data, _, nx, _ = u0.shape
        # timesteps = timesteps + 1
        onedata = False
        if n_data == 1:
            u0 = u0.repeat(2,1,1,1)
            n_data = 2
            onedata = True
        u0 = u0.permute(0,2,3,1).unsqueeze(3) # (n_data, nx, nx, 1, 4)
        pde_params = torch.Tensor([[self.eta, self.zeta]]).repeat(n_data, 1)
        # traj, _ = self.sim(u0,pde_params,self.grid, nt=timesteps) # (n_data, nt, nx, nx, 4)
        # traj, _, _= self.sim(u0, pde_params, self.grid)
        traj, _, _= self.sim.n_step_sim(u0, pde_params, self.grid, 0, n_steps=timesteps)
        # traj, _= self.sim(u0, pde_params, self.grid, 0, n_steps=timesteps)
        traj = traj.permute(0,1,4,2,3)# (n_data, nt, 4, nx, nx)
        if onedata:
            traj = traj[:1]
        return traj




def _evolve_sim(u0, cfg, t0=0, timesteps=1):
    # u0 has shape (n_data, 1, nx, nx)
    device = cfg.device
    batch_size = cfg.generate_data.batch_size

    nt = cfg.generate_data.nt
    end_time = cfg.generate_data.end_time

    if cfg.equation == 'NS':
        dt = round(end_time / (nt - 1) * (130 / (cfg.nt - 1)), 5)
    elif cfg.equation == 'KS':
        dt_want = round(end_time / (nt - 1) * (130 / (cfg.nt - 1)), 5)
        dt = round(end_time / (nt - 1), 5)
        # print(dt_want, dt)
        assert dt_want % dt < 1e-5 and dt_want % dt > -1e-5
        timesteps = int(dt_want / dt * timesteps)
    
    # print(f'dt: {dt}, timesteps: {timesteps}')
    
    if cfg.equation == 'NS':
        vis = cfg.generate_data.vis
        fid = cfg.generate_data.nx
        force = cfg.generate_data.force
        sim = NS(tmax=dt, dt=cfg.generate_data.dt, vis=vis, fid=fid, force=force, device=device)
    elif cfg.equation == 'KS':
        fid = cfg.generate_data.nx
        sim = KS(tmax=dt, fid=fid, device=device)

    u0 = u0.to(device)
    n_data = u0.shape[0]
    solution = []

    for batch_idx, u in enumerate(u0.split(batch_size)):
        try:
            traj = [u]
            for i in range(timesteps):
                traj.append(sim.query_out(traj[-1].squeeze(1)).unsqueeze(1))  # (n_data, 1, nx, nx)
            traj = torch.stack(traj, dim=1)
            traj = traj[:, 1:]  # (n_data, timesteps, 1, nx, nx)
            traj = traj.cpu()
        except AssertionError:
            raise Exception('An error occurred - possibly an underflow.')
        solution.append(traj)

    solution = torch.cat(solution, dim=0)  # (n_data, timesteps, 1, nx, nx)
    return solution  # (n_data, timesteps, 1, nx, nx)

def _evolve_ps(u0, cfg, t0=0, timesteps=1, dt=None):
    # print(u0.max(), u0.min(), u0.shape)
    device = cfg.device
    batch_size = cfg.generate_data.batch_size

    pde = _get_pde_object(cfg)

    pde_string = str(pde)
    # Tolerance of the solver
    atol = cfg.generate_data.atol
    rtol = cfg.generate_data.rtol
    
    nt = cfg.generate_data.nt
    end_time = cfg.generate_data.end_time
    if dt is None:
        dt = end_time / (nt-1) * (130/(cfg.nt-1))
        # dt = end_time / (nt-1)
    nx = pde.grid_size[1]
    
    # The field u, the coordinations (xcoord, tcoord) and dx, dt are saved
    # Only nt_effective time steps of each trajectories are saved
    # h5f_u = dataset.create_dataset(f'pde_{pde.nt_effective}-{nx}', (num_samples, pde.nt_effective, nx), dtype=float)

    # if pde_string == 'Burgers':
    #     u0 = torch.tensor(inv_cole_hopf(u0.cpu().numpy()))

    t = np.linspace(t0, t0+dt*timesteps, timesteps+1)

    assert u0.shape[1] == 1
    u0 = u0.squeeze(1)
    u0 = u0.to(device)
    n_data = u0.shape[0]
    solution = []
    for batch_idx, u in enumerate(u0.split(batch_size)):
        try:
            # T = pde.tmax
            # L = pde.L

            # if pde_string == 'Burgers':
            #     u = inv_cole_hopf(u, pde)
            # t = np.linspace(0, T, nt)
            # x = np.linspace(0, (1 - 1.0 / nx) * L, nx)
        
            spectral_method = pde.pseudospectral_reconstruction_batch
            t = torch.tensor(t).to(device)

            solved_trajectory = odeint(func=spectral_method,
                                        t=t,
                                        y0=u,
                                        method=cfg.generate_data.solver,
                                        atol=atol,
                                        rtol=rtol)
            sol = solved_trajectory.permute(1,0,2) # (n_data, nt, nx)
            sol = sol[:, 1:]

            # if pde_string == 'Burgers':
            #     sol = pde.to_burgers(sol.reshape(-1,nx)).reshape(*sol.shape)
            sol = sol.cpu() # (n_data, nt, nx)
        except AssertionError:
            raise Exception('An error occured - possibly an underflow.')
        solution.append(sol)
    solution = torch.cat(solution, dim=0) # (n_data, nt, nx)
    solution = solution.unsqueeze(2) # (n_data, nt, 1, nx)

    return solution # (n_data, nt, 1, nx)
    
    