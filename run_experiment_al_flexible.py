import h5py
import torch
import numpy as np
from neuralop.models import FNO, TFNO
from tqdm import tqdm
import random
from itertools import islice

import argparse
import time

from eval_utils import compute_metrics
from utils import set_seed, flatten_configdict, trajectory_model, direct_model, split_model, normalized_model, residual_model, normalized_residual_model
from acquisition.acquirer_flexible import Acquirer

from omegaconf import OmegaConf
import hydra
import wandb

from synthetic_acquisition import Y_from_selected, Y_from_selected_cheat

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

class Traj_dataset:
    traj_train_32 = None
    traj_test = None
    pool = None
    pool_with_traj = None

class Pool:
    def __init__(self, path, datasize):
        self.path = path
        self.datasize = datasize
    
    def __getitem__(self, key):
        with h5py.File(self.path, 'r') as f:
            if isinstance(key, torch.Tensor):
                if key.ndim == 0:
                    return torch.tensor(f['train']['pde'][key.item(), 0], dtype=torch.float32)

                # Handle PyTorch tensor
                if key.dtype == torch.bool:
                    # Boolean indexing
                    return torch.tensor(np.stack([f['train']['pde'][i, 0] for i, select in enumerate(key) if select], axis=0), dtype=torch.float32)
                else:
                    # Integer indexing
                    return torch.tensor(np.stack([f['train']['pde'][i.item(), 0] for i in key], axis=0), dtype=torch.float32)
            elif isinstance(key, tuple):
                return torch.tensor(np.stack([f['train']['pde'][k, 0] for k in key], axis=0), dtype=torch.float32)
            elif isinstance(key, slice):
                return torch.tensor(f['train']['pde'][key.start:key.stop:key.step, 0], dtype=torch.float32)
            else:
                assert key < self.datasize
                return torch.tensor(f['train']['pde'][key, 0], dtype=torch.float32)
    
    def __len__(self):
        return self.datasize

class Pool_with_traj:
    def __init__(self, path, timestep, datasize):
        self.path = path
        self.timestep = timestep
        self.datasize = datasize

    def __getitem__(self, key):
        with h5py.File(self.path, 'r') as f:
            if isinstance(key, torch.Tensor):
                if key.ndim == 0:
                    return torch.tensor(f['train']['pde'][key.item(), :131:self.timestep], dtype=torch.float32)

                # Handle PyTorch tensor
                if key.dtype == torch.bool:
                    # Boolean indexing
                    return torch.tensor(np.stack([f['train']['pde'][i, :131:self.timestep] for i, select in enumerate(key) if select], axis=0), dtype=torch.float32)
                else:
                    # Integer indexing
                    return torch.tensor(np.stack([f['train']['pde'][i.item(), :131:self.timestep] for i in key], axis=0), dtype=torch.float32)
            elif isinstance(key, tuple):
                return torch.tensor(np.stack([f['train']['pde'][k, :131:self.timestep] for k in key], axis=0), dtype=torch.float32)
            elif isinstance(key, slice):
                return torch.tensor(f['train']['pde'][key.start:key.stop:key.step, :131:self.timestep], dtype=torch.float32)
            else:
                assert key < self.datasize
                return torch.tensor(f['train']['pde'][key, :131:self.timestep], dtype=torch.float32)
    
    def __len__(self):
        return self.datasize
    


def run_experiment(cfg):
    wandb.define_metric("datasize")

    unrolling = cfg.train.unrolling
    nt = cfg.nt
    L = nt-1
    ensemble_size = cfg.ensemble_size
    num_acquire = cfg.num_acquire
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size
    initial_datasize = cfg.initial_datasize

    data_nt = Traj_dataset.traj_train_32.shape[1]
    timestep = (data_nt - 1) // (cfg.nt - 1) # 10
    num_channels = Traj_dataset.traj_train_32.shape[2]
    assert len(cfg.model.n_modes) == Traj_dataset.traj_train_32.ndim - 3 # number of spatial dimensions

    # Final metrics
    metrics_list = []


    def train(Y, unrolling=0, acquire_step=0, gaussian_noise=0.0):
        # Y as a list of continuous trajectories
        # model = TFNO(n_modes=tuple(cfg.model.n_modes), hidden_channels=64,
        #             in_channels=num_channels, out_channels=num_channels,
        #             factorization='tucker', implementation='factorized', rank=0.05)
        
        model = FNO(n_modes=tuple(cfg.model.n_modes), hidden_channels=64,
                    in_channels=num_channels, out_channels=num_channels)

        model = model.to(device)
        # model = normalized_model(model, Traj_dataset.mean, Traj_dataset.std, Traj_dataset.mean, Traj_dataset.std)
        model = normalized_residual_model(model, Traj_dataset.mean, Traj_dataset.std)

        if len(Y) == 0:
            return model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        inputs, outputs, unrolls = [], [], []
        for traj in Y:
            if len(traj) == 1:  # Skip trajectories with only one point
                continue
            traj_len = len(traj)
            for t in range(traj_len - 1):
                for r in range(unrolling+1):
                    if t + (r + 1) < traj_len:
                        inputs.append(traj[t])
                        outputs.append(traj[t + (r + 1)])
                        unrolls.append(r)

        inputs = torch.stack(inputs, dim=0) # [datasize, 1, nx]
        outputs = torch.stack(outputs, dim=0) # [datasize, 1, nx]
        unrolls = torch.tensor(unrolls) # [datasize]

        iter_per_epoch = len(inputs[unrolls == 0]) // batch_size + 1

        assert inputs.shape[0] == outputs.shape[0]
        assert inputs.shape[1] == num_channels
        print('Datasize:', inputs.shape[0])

        dataset = torch.utils.data.TensorDataset(inputs, outputs, unrolls)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            # for x, y, unroll in islice(dataloader, iter_per_epoch):
            for x, y, unroll in dataloader:
                optimizer.zero_grad()
                x, y, unroll = x.to(device), y.to(device), unroll.to(device)
                x, y, unroll = x[unroll <= epoch], y[unroll <= epoch], unroll[unroll <= epoch]
                if len(x) == 0:
                    continue

                x = x + gaussian_noise * torch.randn_like(x)
                
                # Unroll the model predictions
                for _ in range(unroll.max()):
                    with torch.no_grad():
                        x[unroll > 0] = model(x[unroll > 0])
                        unroll[unroll > 0] -= 1

                pred = model(x)
                loss = criterion(pred, y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            # wandb.log({f'train/loss_{acquire_step}': total_loss})
        return model

    # test: l2, rel_l2, rmse
    @torch.no_grad()
    def test(ensemble):
        X_test = Traj_dataset.traj_test[:,0] # [datasize, 1, nx]
        Y_test = Traj_dataset.traj_test[:,timestep::timestep]

        testset = torch.utils.data.TensorDataset(X_test, Y_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.eval_batch_size, shuffle=False)
        
        preds = []
        for model in ensemble:
            model = trajectory_model(model, L)
            model.eval()
        
            Y_test_pred = []
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    # print(y_pred.shape, y.shape)
                    assert y_pred.shape == y.shape
                    Y_test_pred.append(y_pred.cpu())
                Y_test_pred = torch.cat(Y_test_pred, dim=0)
            preds.append(Y_test_pred)
        preds = torch.stack(preds, dim=0) # [ensemble_size, datasize, L, nx]
        
        metrics = []
        for i in range(ensemble_size):
            metrics_i = compute_metrics(Y_test, preds[i], d=2, device=device)
            metrics_i = torch.stack(metrics_i,dim=0) # [3, datasize]
            l2_i = metrics_i[0]
            quantile_99 = torch.quantile(l2_i, 0.99)
            quantile_95 = torch.quantile(l2_i, 0.95)
            quantile_50 = torch.quantile(l2_i, 0.5)
            metrics_i_mean = metrics_i.mean(dim=1) # [3]
            metrics_i_all = metrics_i_mean.tolist() + [quantile_99.item(), quantile_95.item(), quantile_50.item()] # [6]
            metrics.append(metrics_i_all) # [ensemble_size, 6]
        metrics = torch.tensor(metrics) # [ensemble_size, 6]
        metrics = metrics.mean(dim=0) # [6]

        metrics_list.append(metrics)

        return metrics
    
    def evaluate(ensemble):
        results={}
        results['datasize']=datasize
        metrics = test(ensemble)
        results['test/L2']=(metrics[0].item())
        results['test/Relative_L2']=(metrics[1].item())
        results['test/MAE']=(metrics[2].item())
        results['test/99_L2']=(metrics[3].item())
        results['test/95_L2']=(metrics[4].item())
        results['test/50_L2']=(metrics[5].item())
        print(results)
        wandb.log(results)
        return metrics[2].item() # rmse

    datasize = 0
    train_indices = {} # a dictionary of form {index: S}
    for d in range(initial_datasize):
        train_indices[d] = torch.ones(L, device=device, dtype=torch.bool)
    assert len(train_indices) == initial_datasize

    Y = []

    for d in range(initial_datasize):
        data = Traj_dataset.pool_with_traj[d] # [nt, 1, nx]
        assert data.shape[0] == nt
        assert data.shape[1] == num_channels
        Y.append(data) # [nt, 1, nx]
        datasize += L

    ensemble = [train(Y, unrolling=cfg.train.unrolling, acquire_step=0, gaussian_noise=cfg.train.gaussian_noise) for _ in tqdm(range(ensemble_size))]
    evaluate(ensemble)

    for acquire_step in range(1, num_acquire+1):
        acquirer = Acquirer(ensemble, Traj_dataset.pool, L, train_indices, cfg, Traj_dataset.max, Traj_dataset.min)
        if cfg.exponential_data:
            selected = acquirer.select(L * int(initial_datasize * cfg.exponential_rate ** acquire_step) - int(initial_datasize * cfg.exponential_rate ** (acquire_step-1)))
        else:
            selected = acquirer.select(L * cfg.batch_acquire)

        datasize += sum([selected[i].sum() for i in selected])

        if cfg.cheat == False:
            Y += Y_from_selected(ensemble, selected, Traj_dataset.pool, L, cfg)
        else:
            Y += Y_from_selected_cheat(ensemble, selected, Traj_dataset.pool_with_traj, L, cfg)

        ensemble = [train(Y, unrolling=cfg.train.unrolling, acquire_step=0, gaussian_noise=cfg.train.gaussian_noise) for _ in tqdm(range(ensemble_size))]
        evaluate(ensemble)
    

    metrics_list = torch.stack(metrics_list, dim=0) # [num_acquire+1, 6]
    mean_log_metrics = metrics_list.log().mean(dim=0) # [6]
    results = {}
    results['mean_log_test/L2'] = mean_log_metrics[0].item()
    results['mean_log_test/Relative_L2'] = mean_log_metrics[1].item()
    results['mean_log_test/MAE'] = mean_log_metrics[2].item()
    results['mean_log_test/99_L2'] = mean_log_metrics[3].item()
    results['mean_log_test/95_L2'] = mean_log_metrics[4].item()
    results['mean_log_test/50_L2'] = mean_log_metrics[5].item()
    print(results)
    wandb.log(results)


def mean_std_normalize():
    ndim = Traj_dataset.traj_train_32.ndim
    mean = Traj_dataset.traj_train_32.mean(dim=[i for i in range(ndim) if i != 2], keepdim=True).squeeze(1)
    std = Traj_dataset.traj_train_32.std(dim=[i for i in range(ndim) if i != 2], keepdim=True).squeeze(1)
    print(f'Mean: {mean}, Std: {std}')
    Traj_dataset.mean = mean
    Traj_dataset.std = std

@hydra.main(version_base=None, config_path="cfg_flexible", config_name="config.yaml")
def main(cfg: OmegaConf):
    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    if cfg.wandb.use:
        if cfg.wandb.project is None:
            raise ValueError('cfg.wandb.project must be set if cfg.wandb.use is True')
        wandb.init(
            project=cfg.wandb.project,
            config=flatten_configdict(cfg),
            entity=cfg.wandb.entity,
            # settings=wandb.Settings(start_method="thread"),
        )
    else:
        wandb.init(mode="disabled")
    
    
    print('Loading training data...')
    with h5py.File(cfg.dataset.train_path, 'r') as f:
        Traj_dataset.traj_train_32 = torch.tensor(f['train']['pde'][:32, :131], dtype=torch.float32)
    print('Loading test data...')
    with h5py.File(cfg.dataset.test_path, 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde'][:cfg.testsize, :131], dtype=torch.float32)

    timestep = (Traj_dataset.traj_train_32.shape[1] - 1) // (cfg.nt - 1) # 10

    Traj_dataset.pool = Pool(cfg.dataset.train_path, datasize=cfg.datasize)
    Traj_dataset.pool_with_traj = Pool_with_traj(cfg.dataset.train_path, timestep, datasize=cfg.datasize)

    mean_std_normalize()

    Traj_dataset.max = Traj_dataset.traj_train_32.max().item()
    Traj_dataset.min = Traj_dataset.traj_train_32.min().item()

    run_experiment(cfg)
    wandb.finish()

if __name__ == '__main__':
    main()

