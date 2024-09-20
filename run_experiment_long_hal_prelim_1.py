import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm
import os

from eval_utils import compute_metrics
from utils import set_seed, flatten_configdict, trajectory_model, ensemble_mean_model, split_model, normalized_model, torch_expand

from generate_data import evolve

from omegaconf import OmegaConf
import hydra

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

class Traj_dataset:
    traj_train = None
    traj_valid = None
    traj_test = None

@torch.no_grad()
def create_synthetic_data(cfg):
    device = cfg.device
    nt = cfg.nt
    threshold = cfg.threshold

    datasize = cfg.datasize
    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (cfg.nt - 1) # 10
    max = Traj_dataset.traj_train[:32].max()
    min = Traj_dataset.traj_train[:32].min()
    # assert timestep == 10 # hardcoded for now (130/ (14-1) = 10)

    X = Traj_dataset.traj_train[:datasize,0:1] # [datasize, 1, nx]

    # train_indices = torch.zeros(datasize, nt-1, dtype=torch.bool) # [datasize, nt-1]
    train_indices = torch.ones(datasize, nt-1).bool()
    ready = torch.ones(datasize).bool()

    preds = [X]
    for t in range(nt-1):
        X_t = preds[-1].clone()
        X_t[train_indices[:,t]] = evolve(X_t, cfg)[:,-1:] # [datasize, 1, nx]

        preds.append(X_t)
    preds = torch.cat(preds, dim=1) # [datasize, nt, nx]


    synthetic_data = {'train_indices': train_indices, 'ready': ready, 'Y': preds}

    return synthetic_data

def mean_std_normalize():
    assert Traj_dataset.traj_train is not None
    mean = Traj_dataset.traj_train[:32].mean()
    std = Traj_dataset.traj_train[:32].std()
    print(f'Mean: {mean}, Std: {std}')
    Traj_dataset.mean = mean
    Traj_dataset.std = std

def max_min_normalize():
    assert Traj_dataset.traj_train is not None
    max_val = Traj_dataset.traj_train[:32].max()
    min_val = Traj_dataset.traj_train[:32].min()
    mean = (max_val + min_val) / 2
    std = (max_val - min_val) / 2
    print(f'Max: {max_val}, Min: {min_val}')
    Traj_dataset.mean = mean
    Traj_dataset.std = std

@hydra.main(version_base=None, config_path="cfg_long_hal", config_name="config.yaml")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    prelim_datasize = cfg.prelim_datasize
    p_list = cfg.p_list
    equation = cfg.equation
    path = f'results/long_hal_prelim/{equation}_{cfg.nt}/'

    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    print('Loading training data...')
    with h5py.File(cfg.dataset.train_path, 'r') as f:
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:cfg.datasize+max(cfg.prelim_datasize), :131], dtype=torch.float32)
    print('Loading test data...')
    with h5py.File(cfg.dataset.test_path, 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:cfg.testsize, :131], dtype=torch.float32)

    if cfg.equation == 'Heat' or cfg.equation == 'KS':
        max_min_normalize()
    else:
        mean_std_normalize()
    
    # ensemble, metrics = create_ensemble(datasize, cfg) # list of models
    os.makedirs(path, exist_ok=True)
    # torch.save(ensemble, path + f'{datasize}/ensemble.pt')
    # torch.save(metrics, path + f'{datasize}/metrics.pt')
    print(f'Creating synthetic data...')
    synthetic_data = create_synthetic_data(cfg)
    if cfg.prelim_save:
        torch.save(synthetic_data, path + f'/synthetic_data_p1.0.pt')

if __name__ == '__main__':
    main()

