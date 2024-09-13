import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm
import random

import argparse
import time

from eval_utils import compute_metrics

from utils import set_seed, flatten_configdict
from acquisition.acquirers import select, select_time

from omegaconf import OmegaConf
import hydra
import wandb

class Traj_dataset:
    traj_train = None
    traj_valid = None
    traj_test = None


def run_experiment(cfg):
    wandb.define_metric("datasize")
    # logarithmic scale
    wandb.define_metric("l2", step_metric="datasize")
    wandb.define_metric("rel_l2", step_metric="datasize")
    wandb.define_metric("mse", step_metric="datasize")

    unrolling = cfg.train.unrolling
    nt = cfg.nt
    ensemble_size = cfg.ensemble_size
    selection_method = cfg.selection_method
    # batch_acquire = cfg.batch_acquire
    num_acquire = cfg.num_acquire
    initial_time_steps = cfg.initial_time_steps
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size

    def train(Y, train_nts, **kwargs):
        assert unrolling == 0

        acquire_step = kwargs.get('acquire_step', 0)

        model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        inputs = []
        outputs = []
        for b in range(Y.shape[0]):
            for t in range(train_nts[b].item()-1):
                inputs.append(Y[b,t])
                outputs.append(Y[b, t+1])
        inputs = torch.stack(inputs, dim=0).unsqueeze(1)
        outputs = torch.stack(outputs, dim=0).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(inputs, outputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            model.train()
            # max_unrolling = epoch if epoch <= unrolling else unrolling
            # unrolling_list = [r for r in range(max_unrolling + 1)]

            total_loss = 0
            for x, y in dataloader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                
                pred = model(x)
                loss = criterion(pred, y)

                # loss = torch.sqrt(loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            # wandb.log({f'train/loss_{acquire_step}': total_loss})
        return model

    def test(model):
        X_test = Traj_dataset.traj_test[:,0,:].unsqueeze(1).to(device)
        Y_test = Traj_dataset.traj_test[:,-1,:].unsqueeze(1).to(device)

        testset = torch.utils.data.TensorDataset(X_test, Y_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        model.eval()
    
        Y_test_pred = []
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                Y_test_pred.append(model(x))
            Y_test_pred = torch.cat(Y_test_pred, dim=0).to(device)
        
        metrics = compute_metrics(Y_test, Y_test_pred, d=1)

        return metrics
    
    def test_trajectory(model):
        X_test = Traj_dataset.traj_test[:,0].unsqueeze(1).to(device)
        Y_test = Traj_dataset.traj_test[:,timestep::timestep].to(device)

        testset = torch.utils.data.TensorDataset(X_test, Y_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        model.eval()
    
        Y_test_pred = []
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                # print(y_pred.shape, y.shape)
                assert y_pred.shape == y.shape
                Y_test_pred.append(y_pred)
            Y_test_pred = torch.cat(Y_test_pred, dim=0).to(device)
        
        metrics = compute_metrics(Y_test, Y_test_pred, d=2)

        return metrics

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
            return torch.cat(trajectory, dim=1) # [cfg.train.batch_size, unrolling, nx]

    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    assert timestep == 10 # hardcoded for now (130/ (14-1) = 10)

    X = Traj_dataset.traj_train[:,0].unsqueeze(1).to(device)
    Y = Traj_dataset.traj_train[:,0::timestep].to(device)

    train_nts = torch.ones(X.shape[0], device=device, dtype=torch.int64)
    # values are between 1 and 14, inclusive
    # 1 means only initial data, 14 means all data

    train_nts[:initial_time_steps//(nt-1)] = nt
    if initial_time_steps % (nt-1) != 0:
        train_nts[initial_time_steps//(nt-1)] = initial_time_steps % (nt-1)

    # train_idxs = torch.arange(initial_datasize, device=device)
    # pool_idxs = torch.arange(initial_datasize, X.shape[0], device=device)

    # X_train = X[train_idxs]
    # Y_train = Y[train_idxs]

    # X_pool = X[pool_idxs]

    ensemble = [train(Y, train_nts, acquire_step=0) for _ in tqdm(range(ensemble_size))]

    results = {'datasize': [], 'l2': [], 'rel_l2': [], 'mse': []}

    results['datasize'].append((train_nts-1).sum().item())
    # rel_l2_list = [test(direct_model(model, nt-1))[1].mean().item() for model in ensemble]
    metrics_list = torch.stack([torch.stack(test_trajectory(trajectory_model(model, nt-1))) for model in ensemble]) # [ensemble_size, 3, datasize]
    results['l2'].append(metrics_list[:, 0, :].mean().item())
    results['rel_l2'].append(metrics_list[:, 1, :].mean().item())
    results['mse'].append(metrics_list[:, 2, :].mean().item())
    print(f'Datasize: {results["datasize"][-1]}, L2: {results["l2"][-1]}, Rel_l2: {results["rel_l2"][-1]}, MSE: {results["mse"][-1]}')

    # wandb.log({'datasize': results['datasize'][-1], f'rel_l2_{acquire_step}': results['rel_l2'][-1], f'rel_l2_trajectory_{acquire_step}': results['rel_l2_trajectory'][-1]})
    wandb.log({'datasize': results['datasize'][-1], f'l2': results['l2'][-1], f'rel_l2': results['rel_l2'][-1], f'mse': results['mse'][-1]})
    
    for acquire_step in range(1, num_acquire+1):
        train_nts = select_time(ensemble, Y, train_nts, (train_nts-1).sum().item(), selection_method=selection_method, mode=cfg.acquisition_mode, device=device)

        X = Traj_dataset.traj_train[:,0].unsqueeze(1).to(device)
        Y = Traj_dataset.traj_train[:,0::timestep].to(device)

        ensemble = [train(Y, train_nts, acquire_step=acquire_step) for _ in tqdm(range(ensemble_size))]

        results['datasize'].append((train_nts-1).sum().item())
        metrics_list = torch.stack([torch.stack(test_trajectory(trajectory_model(model, nt-1))) for model in ensemble]) # [ensemble_size, 3, datasize]
        results['l2'].append(metrics_list[:, 0, :].mean().item())
        results['rel_l2'].append(metrics_list[:, 1, :].mean().item())
        results['mse'].append(metrics_list[:, 2, :].mean().item())
        print(f'Datasize: {results["datasize"][-1]}, L2: {results["l2"][-1]}, Rel_l2: {results["rel_l2"][-1]}, MSE: {results["mse"][-1]}')
        wandb.log({'datasize': results['datasize'][-1], f'l2': results['l2'][-1], f'rel_l2': results['rel_l2'][-1], f'mse': results['mse'][-1]})
    
    return results

@hydra.main(version_base=None, config_path="cfg_time", config_name="config.yaml")
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
    with h5py.File(f'data_large/{cfg.equation}_train_100000_default.h5', 'r') as f:
        # Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:10000, :131], dtype=torch.float32, device=cfg.device)
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:cfg.datasize, :131], dtype=torch.float32)
        # Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:100, :131], dtype=torch.float32, device=cfg.device)
    # print('Loading validation data...')
    # with h5py.File(f'data_large/{cfg.equation}_valid_1024_default.h5', 'r') as f:
    #     Traj_dataset.traj_valid = torch.tensor(f['valid']['pde_140-256'][:, :131], dtype=torch.float32)
    print('Loading test data...')
    with h5py.File(f'data_large/{cfg.equation}_test_100000_default.h5', 'r') as f:
        # Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:, :131], dtype=torch.float32)
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:10000, :131], dtype=torch.float32)

    run_experiment(cfg)

    wandb.finish()

if __name__ == '__main__':
    main()

