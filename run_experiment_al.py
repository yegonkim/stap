import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm
import random
import copy

import argparse
import time

from eval_utils import compute_metrics
from custom_paths import get_results_path
from utils import set_seed, flatten_configdict
from acquisition.acquirers import select

from omegaconf import OmegaConf
import hydra
import wandb

class Traj_dataset:
    traj_train = None
    traj_valid = None
    traj_test = None


def run_experiment(cfg):
    # unrolling = cfg.train.unrolling
    nt = cfg.nt
    ensemble_size = cfg.ensemble_size
    selection_method = cfg.selection_method
    num_acquire = cfg.num_acquire
    initial_datasize = cfg.initial_datasize
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size
    batch_acquire = cfg.batch_acquire
    num_random_pool = cfg.num_random_pool

    def train(model, X_train, Y_train, **kwargs):
        acquire_step = kwargs.get('acquire_step', 0)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        return model

    def test(model):
        X_test = Traj_dataset.traj_test[:,0].unsqueeze(1).to(device)
        Y_test = Traj_dataset.traj_test[:,timestep].unsqueeze(1).to(device)

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
    
    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    assert timestep == 10

    X = Traj_dataset.traj_train[:,0].unsqueeze(1).to(device)
    Y = Traj_dataset.traj_train[:,timestep].unsqueeze(1).to(device)

    train_idxs = torch.arange(initial_datasize, device=device)
    pool_idxs = torch.arange(initial_datasize, X.shape[0], device=device)

    X_train = X[train_idxs]
    Y_train = Y[train_idxs]

    X_pool = X[pool_idxs]

    if cfg.fix_initial_weights:
        initial_ensemble = [FNO(n_modes=(256, ), hidden_channels=64, in_channels=1, out_channels=1) for _ in range(ensemble_size)]

    if cfg.fix_initial_weights:
        ensemble = [train(copy.deepcopy(model), X_train, Y_train, acquire_step=0) for model in initial_ensemble]
    else:
        ensemble = [FNO(n_modes=(256, ), hidden_channels=64, in_channels=1, out_channels=1) for _ in range(ensemble_size)]
        ensemble = [train(model, X_train, Y_train, acquire_step=0) for model in ensemble]

    results = {'datasize': [], 'rel_l2': [], 'rel_l2_trajectory': []}

    acquire_step=0
    results['datasize'].append(train_idxs.shape[0])
    rel_l2_list = [test(model)[1].mean().item() for model in ensemble]
    results['rel_l2'].append(torch.mean(torch.tensor(rel_l2_list)).item())
    print(f'Datasize: {results["datasize"][-1]}, Rel_l2: {results["rel_l2"][-1]}')

    wandb.log({'datasize': results['datasize'][-1], f'rel_l2': results['rel_l2'][-1]})
    
    for acquire_step in range(1, num_acquire+1):
        # new_idxs = select(ensemble, X_train, X_pool, train_idxs.shape[0], selection_method=selection_method, device=device)
        new_idxs = select(ensemble, X_train, X_pool, batch_acquire, selection_method=selection_method, device=device, num_random_pool=num_random_pool)
        # new_idxs = select_var(ensemble, X_pool, batch_acquire)

        new_idxs = new_idxs.to(device)
        # print(new_idxs)
        # print(f'{len(new_idxs)=}')
        logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool, device=device)
        logical_new_idxs[new_idxs] = True
        train_idxs = torch.cat([train_idxs, pool_idxs[logical_new_idxs]], dim=-1)
        pool_idxs = pool_idxs[~logical_new_idxs]

        X_train = X[train_idxs]
        Y_train = Y[train_idxs]

        X_pool = X[pool_idxs]

        # ensemble = [train(X_train, Y_train) for _ in tqdm(range(ensemble_size))]

        if cfg.fix_initial_weights:
            ensemble = [train(copy.deepcopy(model), X_train, Y_train, acquire_step=0) for model in initial_ensemble]
        else:
            ensemble = [FNO(n_modes=(256, ), hidden_channels=64, in_channels=1, out_channels=1) for _ in range(ensemble_size)]
            ensemble = [train(model, X_train, Y_train, acquire_step=0) for model in ensemble]

        results['datasize'].append(train_idxs.shape[0])
        rel_l2_list = [test(model)[1].mean().item() for model in ensemble]
        results['rel_l2'].append(torch.mean(torch.tensor(rel_l2_list)).item())
        print(f'Datasize: {results["datasize"][-1]}, Rel_l2: {results["rel_l2"][-1]}')

        wandb.log({'datasize': results['datasize'][-1], f'rel_l2': results['rel_l2'][-1]})
    return results

@hydra.main(version_base=None, config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    print('Seed:', cfg.seed)

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
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:cfg.datasize, :131], dtype=torch.float32)
    print('Loading test data...')
    with h5py.File(cfg.dataset.test_path, 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:cfg.testsize, :131], dtype=torch.float32)

    # with h5py.File(f'data/{cfg.equation}_train_1024_default.h5', 'r') as f:
    #     Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    # with h5py.File(f'data/{cfg.equation}_valid_1024_default.h5', 'r') as f:
    #     Traj_dataset.traj_valid = torch.tensor(f['valid']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    # with h5py.File(f'data/{cfg.equation}_test_4096_default.h5', 'r') as f:
    #     Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:], dtype=torch.float32)[:, :131]

    results = run_experiment(cfg)

    wandb.finish()

    # print(results)
    # save_path = get_results_path() + f'/results_al_{args.equation}_{args.experiment}_{args.batch_acquire}_{args.batch_acquire}_{args.batch_acquire}_{time.strftime("%Y%m%d-%H%M%S")}.pt'
    # torch.save(results, save_path)
    # print(f'Results saved to {save_path}')

if __name__ == '__main__':
    main()

