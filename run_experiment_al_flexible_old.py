import h5py
import torch
import numpy as np
from neuralop.models import FNO, TFNO
from tqdm import tqdm
import random

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
    # logarithmic scale
    wandb.define_metric("l2", step_metric="datasize")
    wandb.define_metric("rel_l2", step_metric="datasize")
    wandb.define_metric("mse", step_metric="datasize")
    wandb.define_metric("l2_trajectory", step_metric="datasize")
    wandb.define_metric("rel_l2_trajectory", step_metric="datasize")
    wandb.define_metric("mse_trajectory", step_metric="datasize")

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

    def train(Y, unrolling=0, acquire_step=0):
        assert unrolling == 0
        # Y as a list of continuous trajectories
        # model = TFNO(n_modes=tuple(cfg.model.n_modes), hidden_channels=64,
        #             in_channels=num_channels, out_channels=num_channels,
        #             factorization='tucker', implementation='factorized', rank=0.05)
        
        model = FNO(n_modes=tuple(cfg.model.n_modes), hidden_channels=64,
                    in_channels=num_channels, out_channels=num_channels)

        model = model.to(device)
        model = normalized_model(model, Traj_dataset.mean, Traj_dataset.std, Traj_dataset.mean, Traj_dataset.std)
        # model = normalized_residual_model(model, Traj_dataset.mean, Traj_dataset.std)

        if len(Y) == 0:
            return model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        inputs, outputs = [], []
        for traj in Y:
            if len(traj) == 1: # skip the trajectory with only one point
                continue
            inputs.append(traj[:-1])
            outputs.append(traj[1:])
        inputs = torch.cat(inputs, dim=0)
        outputs = torch.cat(outputs, dim=0)
        assert inputs.shape[0] == outputs.shape[0]
        assert inputs.shape[1] == num_channels
        print('Datasize:', inputs.shape[0])
        
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


    # test with teacher forcing
    @torch.no_grad()
    def test_tf(ensemble):
        X_test = Traj_dataset.traj_test[:,0:(nt-1)*timestep:timestep] # [datasize, L, nx]
        Y_test = Traj_dataset.traj_test[:,timestep:nt*timestep:timestep] # [datasize, L, nx]
        X_test = X_test.flatten(0, 1) # [datasize*L, 1, nx]
        Y_test = Y_test.flatten(0, 1) # [datasize*L, 1, nx]
        # X_test = Traj_dataset.traj_test[:,0].unsqueeze(1).to(device)
        # Y_test = Traj_dataset.traj_test[:,timestep::timestep].to(device)

        testset = torch.utils.data.TensorDataset(X_test, Y_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.eval_batch_size, shuffle=False)

        preds = []
        for model in ensemble:
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
        preds = torch.stack(preds, dim=0) # [ensemble_size, datasize*L, 1, nx]
        
        mean_pred = preds.mean(dim=0) # [datasize*L, 1, nx]
        metrics = compute_metrics(Y_test, mean_pred, d=2, device=device) # (l2, rel_l2, mse)
        metrics = torch.stack(metrics,dim=0).mean(dim=1) # [3]
        metrics[2] = torch.sqrt(metrics[2]) # rmse
        metrics_ensemble = metrics

        metrics_individual = []
        for i in range(ensemble_size):
            metrics_i = compute_metrics(Y_test, preds[i], d=2, device=device)
            metrics_i = torch.stack(metrics_i,dim=0).mean(dim=1) # [3]
            metrics_i[2] = torch.sqrt(metrics_i[2])
            metrics_individual.append(metrics_i)
        metrics_individual = torch.stack(metrics_individual, dim=0) # [ensemble_size, 3]
        metrics_individual = metrics_individual.mean(dim=0) # [3]

        return metrics_individual, metrics_ensemble

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
        
        mean_pred = preds.mean(dim=0) # [datasize*L, 1, nx]
        metrics = compute_metrics(Y_test, mean_pred, d=2, device=device) # (l2, rel_l2, mse)
        metrics = torch.stack(metrics,dim=0).mean(dim=1) # [3]
        metrics[2] = torch.sqrt(metrics[2]) # rmse
        metrics_ensemble = metrics

        metrics_individual = []
        for i in range(ensemble_size):
            metrics_i = compute_metrics(Y_test, preds[i], d=2, device=device)
            metrics_i = torch.stack(metrics_i,dim=0).mean(dim=1) # [3]
            metrics_i[2] = torch.sqrt(metrics_i[2])
            metrics_individual.append(metrics_i)
        metrics_individual = torch.stack(metrics_individual, dim=0) # [ensemble_size, 3]
        metrics_individual = metrics_individual.mean(dim=0) # [3]

        return metrics_individual, metrics_ensemble

    def evaluate(ensemble):
        results={}
        results['datasize']=datasize
        metrics, metrics_ensemble = test(ensemble)
        results['test/L2']=(metrics[0].item())
        results['test/Relative_L2']=(metrics[1].item())
        results['test/RMSE']=(metrics[2].item())
        results['test_ensemble/L2']=(metrics_ensemble[0].item())
        results['test_ensemble/Relative_L2']=(metrics_ensemble[1].item())
        results['test_ensemble/RMSE']=(metrics_ensemble[2].item())
        metrics, metrics_ensemble = test_tf(ensemble)
        results['test_tf/L2']=(metrics[0].item())
        results['test_tf/Relative_L2']=(metrics[1].item())
        results['test_tf/RMSE']=(metrics[2].item())
        results['test_tf_ensemble/L2']=(metrics_ensemble[0].item())
        results['test_tf_ensemble/Relative_L2']=(metrics_ensemble[1].item())
        results['test_tf_ensemble/RMSE']=(metrics_ensemble[2].item())
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

    ensemble = [train(Y, unrolling=cfg.train.unrolling, acquire_step=0) for _ in tqdm(range(ensemble_size))]
    evaluate(ensemble)

    for acquire_step in range(1, num_acquire+1):
        acquirer = Acquirer(ensemble, Traj_dataset.pool, L, train_indices, cfg)
        if cfg.exponential_data:
            selected = acquirer.select(L * int(initial_datasize * cfg.exponential_rate ** acquire_step) - int(initial_datasize * cfg.exponential_rate ** (acquire_step-1)))
        else:
            selected = acquirer.select(L * cfg.batch_acquire)

        datasize += sum([selected[i].sum() for i in selected])

        if cfg.cheat == False:
            Y += Y_from_selected(ensemble, selected, Traj_dataset.pool, L, cfg)
        else:
            # Y += Y_from_selected_cheat(ensemble, selected, Traj_dataset.pool, L, cfg)
            Y += Y_from_selected_cheat(ensemble, selected, Traj_dataset.pool_with_traj, L, cfg)

        ensemble = [train(Y, unrolling=cfg.train.unrolling, acquire_step=0) for _ in tqdm(range(ensemble_size))]
        evaluate(ensemble)

    # # save train_nts to wandb
    # train_nts = acquirer.train_nts.cpu().numpy()
    # run_dir = wandb.run.dir
    # np.save(os.path.join(run_dir, 'train_nts.npy'), train_nts)
    # artifact = wandb.Artifact(name = "results", type = "dataset")
    # artifact.add_file(os.path.join(run_dir, 'train_nts.npy'))
    # wandb.log_artifact(artifact)

def mean_std_normalize():
    ndim = Traj_dataset.traj_train_32.ndim
    mean = Traj_dataset.traj_train_32.mean(dim=[i for i in range(ndim) if i != 2], keepdim=True).squeeze(1)
    std = Traj_dataset.traj_train_32.std(dim=[i for i in range(ndim) if i != 2], keepdim=True).squeeze(1)
    print(f'Mean: {mean}, Std: {std}')
    Traj_dataset.mean = mean
    Traj_dataset.std = std

def max_min_normalize():
    ndim = Traj_dataset.traj_train_32.ndim
    max_val = Traj_dataset.traj_train_32.max()
    min_val = Traj_dataset.traj_train_32.min()
    mean = (max_val + min_val) / 2
    std = (max_val - min_val) / 2
    print(f'Max: {max_val}, Min: {min_val}')
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

    if cfg.equation == 'Heat' or cfg.equation == 'KS':
        max_min_normalize()
    else:
        mean_std_normalize()

    # mean_std_normalize()

    run_experiment(cfg)
    wandb.finish()

if __name__ == '__main__':
    main()

