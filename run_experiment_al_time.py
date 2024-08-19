import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm
import random

import argparse
import time

from eval_utils import compute_metrics
from custom_paths import get_results_path
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
    unrolling = cfg.train.unrolling
    nt = cfg.nt
    ensemble_size = cfg.ensemble_size
    selection_method = cfg.selection_method
    selection_feature = cfg.selection_feature
    batch_acquire = cfg.batch_acquire
    num_acquire = cfg.num_acquire
    initial_datasize = cfg.initial_datasize
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size

    # def train(X, Y, train_nts, **kwargs):
    #     assert unrolling == 0
        
    #     acquire_step = kwargs.get('acquire_step', 0)

    #     model = FNO(n_modes=(256, ), hidden_channels=64,
    #                 in_channels=1, out_channels=1)

    #     model = model.to(device)

    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    #     criterion = torch.nn.MSELoss()

    #     mask = train_nts > 0
    #     X = X[mask]
    #     Y = Y[mask]
    #     train_nts = train_nts[mask]

    #     dataset = torch.utils.data.TensorDataset(X, Y, train_nts)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #     model.train()
    #     for epoch in range(epochs):
    #         model.train()
    #         max_unrolling = epoch if epoch <= unrolling else unrolling
    #         unrolling_list = [r for r in range(max_unrolling + 1)]

    #         # Loop over every epoch as often as the number of timesteps in one trajectory.
    #         # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    #         # Therefore in expectation the whole available training information is covered.
    #         total_loss = 0
    #         for i in range(nt):
    #             for x, y, nts in dataloader:
    #                 optimizer.zero_grad()
    #                 x, y = x.to(device), y.to(device) # y has shape [cfg.train.batch_size, nt, nx]

    #                 unrolled = random.choice(unrolling_list)
    #                 bs = x.shape[0]

    #                 # steps = [t for t in range(0, nt - 1 - unrolled)]
    #                 # random_steps = random.choices(steps, k=bs)
    #                 assert torch.all(nts - unrolled - 1 > 0)
    #                 random_steps = torch.floor((nts - unrolled) * torch.rand(bs)).to(dtype=torch.int64)

    #                 inputs = torch.stack([y[b, random_steps[b]] for b in range(bs)], dim=0).unsqueeze(1)
    #                 outputs = torch.stack([y[b, random_steps[b] + unrolled+1] for b in range(bs)], dim=0).unsqueeze(1)

    #                 # pushforward
    #                 with torch.no_grad():
    #                     model.eval()
    #                     for _ in range(unrolled):
    #                         inputs = model(inputs)
    #                     model.train()
                    
    #                 pred = model(inputs)
    #                 loss = criterion(pred, outputs)

    #                 # loss = torch.sqrt(loss)
    #                 loss.backward()
    #                 optimizer.step()
    #                 total_loss += loss.item()
    #         scheduler.step()
    #         wandb.log({f'train/loss_{acquire_step}': total_loss})
    #     return model
    
    def train(Y, train_nts, **kwargs):
        assert unrolling == 0

        acquire_step = kwargs.get('acquire_step', 0)

        model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        data = []
        for b in range(Y.shape[0]):
            for t in range(train_nts[b].item()):
                data.append((Y, ))

        dataset = torch.utils.data.TensorDataset(X, Y, train_nts)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            model.train()
            max_unrolling = epoch if epoch <= unrolling else unrolling
            unrolling_list = [r for r in range(max_unrolling + 1)]

            # Loop over every epoch as often as the number of timesteps in one trajectory.
            # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
            # Therefore in expectation the whole available training information is covered.
            total_loss = 0
            for i in range(nt):
                for x, y, nts in dataloader:
                    optimizer.zero_grad()
                    x, y = x.to(device), y.to(device) # y has shape [cfg.train.batch_size, nt, nx]

                    unrolled = random.choice(unrolling_list)
                    bs = x.shape[0]

                    # steps = [t for t in range(0, nt - 1 - unrolled)]
                    # random_steps = random.choices(steps, k=bs)
                    assert torch.all(nts - unrolled - 1 > 0)
                    random_steps = torch.floor((nts - unrolled) * torch.rand(bs)).to(dtype=torch.int64)

                    inputs = torch.stack([y[b, random_steps[b]] for b in range(bs)], dim=0).unsqueeze(1)
                    outputs = torch.stack([y[b, random_steps[b] + unrolled+1] for b in range(bs)], dim=0).unsqueeze(1)

                    # pushforward
                    with torch.no_grad():
                        model.eval()
                        for _ in range(unrolled):
                            inputs = model(inputs)
                        model.train()
                    
                    pred = model(inputs)
                    loss = criterion(pred, outputs)

                    # loss = torch.sqrt(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            scheduler.step()
            wandb.log({f'train/loss_{acquire_step}': total_loss})
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
        X_test = Traj_dataset.traj_test[:,0,:].unsqueeze(1).to(device)
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

    train_nts[:initial_datasize] = nt
    # train_idxs = torch.arange(initial_datasize, device=device)
    # pool_idxs = torch.arange(initial_datasize, X.shape[0], device=device)

    # X_train = X[train_idxs]
    # Y_train = Y[train_idxs]

    # X_pool = X[pool_idxs]

    ensemble = [train(X, Y, train_nts, acquire_step=0) for _ in tqdm(range(ensemble_size))]

    results = {'datasize': [], 'l2': [], 'rel_l2': [], 'mse': []}

    results['datasize'].append(train_idxs.shape[0])
    # rel_l2_list = [test(direct_model(model, nt-1))[1].mean().item() for model in ensemble]
    metrics_list = torch.tensor([test_trajectory(trajectory_model(model, nt-1)) for model in ensemble]) # [ensemble_size, 3, datasize]
    results['l2'].append(metrics_list[:, 0].mean().item())
    results['rel_l2'].append(metrics_list[:, 1].mean().item())
    results['mse'].append(metrics_list[:, 2].mean().item())
    print(f'Datasize: {results["datasize"][-1]}, L2: {results["l2"][-1]}, Rel_l2: {results["rel_l2"][-1]}, MSE: {results["mse"][-1]}')

    # wandb.log({'datasize': results['datasize'][-1], f'rel_l2_{acquire_step}': results['rel_l2'][-1], f'rel_l2_trajectory_{acquire_step}': results['rel_l2_trajectory'][-1]})
    wandb.log({'datasize': results['datasize'][-1], f'l2': results['l2'][-1], f'rel_l2': results['rel_l2'][-1], f'mse': results['mse'][-1]})
    
    for acquire_step in range(1, num_acquire+1):
        new_idxs = select_time(ensemble, Y, train_nts, batch_acquire, selection_method=selection_method, device=device)
        # new_idxs = select_var(ensemble, X_pool, batch_acquire)

        new_idxs = new_idxs.to(device)
        # print(new_idxs)
        # print(f'{len(new_idxs)=}')
        logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool, device=device)
        logical_new_idxs[new_idxs] = True
        train_idxs = torch.cat([train_idxs, pool_idxs[logical_new_idxs]], dim=-1)
        pool_idxs = pool_idxs[~logical_new_idxs]

        X = Traj_dataset.traj_train[:,0].unsqueeze(1).to(device)
        Y = Traj_dataset.traj_train[:,0::timestep].to(device)

        train_nts = torch.zeros(X.shape[0], device=device, dtype=torch.int64)

        ensemble = [train(X, Y, train_nts, acquire_step=0) for _ in tqdm(range(ensemble_size))]

        results['datasize'].append(train_idxs.shape[0])
        metrics_list = torch.tensor([test_trajectory(trajectory_model(model, nt-1)) for model in ensemble]) # [ensemble_size, 3, datasize]
        results['l2'].append(metrics_list[:, 0].mean().item())
        results['rel_l2'].append(metrics_list[:, 1].mean().item())
        results['mse'].append(metrics_list[:, 2].mean().item())
        print(f'Datasize: {results["datasize"][-1]}, L2: {results["l2"][-1]}, Rel_l2: {results["rel_l2"][-1]}, MSE: {results["mse"][-1]}')

        # wandb.log({'datasize': results['datasize'][-1], f'rel_l2_{acquire_step}': results['rel_l2'][-1], f'rel_l2_trajectory_{acquire_step}': results['rel_l2_trajectory'][-1]})
        wandb.log({'datasize': results['datasize'][-1], f'rel_l2': results['rel_l2'][-1], f'rel_l2_trajectory': results['rel_l2_trajectory'][-1]})
    
    return results

@hydra.main(version_base=None, config_path="cfg_time", config_name="config.yaml")
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

    with h5py.File(f'data/{cfg.equation}_train_1024_default.h5', 'r') as f:
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    with h5py.File(f'data/{cfg.equation}_valid_1024_default.h5', 'r') as f:
        Traj_dataset.traj_valid = torch.tensor(f['valid']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    with h5py.File(f'data/{cfg.equation}_test_4096_default.h5', 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:], dtype=torch.float32)[:, :131]

    results = run_experiment(cfg)

    wandb.finish()

    # print(results)
    # save_path = get_results_path() + f'/results_al_{args.equation}_{args.experiment}_{args.batch_acquire}_{args.batch_acquire}_{args.batch_acquire}_{time.strftime("%Y%m%d-%H%M%S")}.pt'
    # torch.save(results, save_path)
    # print(f'Results saved to {save_path}')

if __name__ == '__main__':
    main()

