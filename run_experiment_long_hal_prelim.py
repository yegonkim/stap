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


def create_ensemble(datasize, cfg):
    print('Creating model...')
    device = cfg.device
    
    unrolling = cfg.train.unrolling
    nt = cfg.nt
    ensemble_size = cfg.prelim_ensemble_size
    # fix ensemble size to 1 for now
    # ensemble_size = 5
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size

    def train(Y, train_indices, **kwargs):
        model = FNO(n_modes=tuple(cfg.model.n_modes), hidden_channels=64,
                    in_channels=1, out_channels=1)
        model = model.to(device)
        model = normalized_model(model, Traj_dataset.mean, Traj_dataset.std, Traj_dataset.mean, Traj_dataset.std)

        if train_indices.sum().item() == 0:
            return model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        inputs = Y[:,:-1][train_indices][:,None,:]
        outputs = Y[:,1:][train_indices][:,None,:]
        assert inputs.shape[0] == train_indices.sum().item()
        
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
    

    def test_per_trajectory(model):
        X_test = Traj_dataset.traj_test[:,0].unsqueeze(1)
        Y_test = Traj_dataset.traj_test[:,timestep::timestep]

        testset = torch.utils.data.TensorDataset(X_test, Y_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.eval_batch_size, shuffle=False)

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
        
        metrics = compute_metrics(Y_test, Y_test_pred, d=2)

        return metrics


    def evaluate(ensemble):
        results = {'l2_per_model': [], 'mse_per_model': [], 'rel_l2_per_model': [], 'l2_timestep': [], 'mse_per_timestep': [], 'rel_l2_timestep': [], 'l2_trajectory': [], 'mse_per_trajectory': [], 'rel_l2_trajectory': []}
        metrics_list = torch.stack([torch.stack(test_per_trajectory(trajectory_model(model, nt-1))) for model in ensemble]) # [ensemble_size, 3, datasize]
        results['l2_per_model'].append(metrics_list[:, 0, :].mean().item())
        results['rel_l2_per_model'].append(metrics_list[:, 1, :].mean().item())
        results['mse_per_model'].append(metrics_list[:, 2, :].mean().item())
        metrics_list = torch.stack(test_per_trajectory(trajectory_model(ensemble_mean_model(ensemble), nt-1))) # [3, datasize]
        results['l2_timestep'].append(metrics_list[0, :].mean().item())
        results['rel_l2_timestep'].append(metrics_list[1, :].mean().item())
        results['mse_per_timestep'].append(metrics_list[2, :].mean().item())
        metrics_list = torch.stack(test_per_trajectory(ensemble_mean_model([trajectory_model(model, nt-1) for model in ensemble]))) # [3, datasize]
        results['l2_trajectory'].append(metrics_list[0, :].mean().item())
        results['rel_l2_trajectory'].append(metrics_list[1, :].mean().item())
        results['mse_per_trajectory'].append(metrics_list[2, :].mean().item())
        return results
    
    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    # assert timestep == 10 # hardcoded for now (130/ (14-1) = 10)
    print(f'Timestep: {timestep}')


    X = Traj_dataset.traj_train[:,0].unsqueeze(1)
    Y = Traj_dataset.traj_train[:,0::timestep]

    train_indices = torch.zeros(X.shape[0], nt-1, dtype=torch.bool)
    train_indices[-datasize:, :] = True

    ensemble = [train(Y, train_indices, acquire_step=0) for _ in tqdm(range(ensemble_size))]
    results = evaluate(ensemble)

    return ensemble, results

@torch.no_grad()
def create_synthetic_data(ensemble, p, cfg):
    device = cfg.device
    nt = cfg.nt
    threshold = cfg.threshold

    for model in ensemble:
        model.eval()

    datasize = cfg.datasize
    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (cfg.nt - 1) # 10
    max = Traj_dataset.traj_train[:32].max()
    min = Traj_dataset.traj_train[:32].min()
    # assert timestep == 10 # hardcoded for now (130/ (14-1) = 10)

    X = Traj_dataset.traj_train[:datasize,0:1] # [datasize, 1, nx]

    # train_indices = torch.zeros(datasize, nt-1, dtype=torch.bool) # [datasize, nt-1]
    train_indices = torch.bernoulli(torch.ones(datasize, nt-1) * p).bool()

    # if mode in ['single', 'ensemble']:
    #     preds = [X]
    #     for t in range(nt-1):
    #         X_t = preds[-1].clone()
    #         if (train_indices[:,t] == True).any():
    #             print('simulation')
    #             X_t[train_indices[:,t]] = evolve(X_t[train_indices[:,t]], cfg)[:,-1:] # [datasize, 1, nx]
    #         if (train_indices[:,t] == False).any():
    #             print('prediction')
    #             if mode == 'single':
    #                 model = ensemble[0]
    #                 X_t[~train_indices[:,t]] = split_model(model, cfg.eval_batch_size)(X_t[~train_indices[:,t]].to(device)).cpu() # [datasize, 1, nx]
    #             elif mode == 'ensemble':
    #                 X_t[~train_indices[:,t]] = split_model(ensemble_mean_model(ensemble), cfg.eval_batch_size)(X_t[~train_indices[:,t]].to(device)).cpu() # [datasize, 1, nx]
    #         preds.append(X_t)
    #     preds = torch.cat(preds, dim=1) # [datasize, nt, nx]
    # elif mode in ['ensemble_whole']:
    #     preds = [torch_expand(X[:,None], 1, len(ensemble))] # [datasize, ensemble_size, 1, nx]
    #     for t in range(nt-1):
    #         X_t = preds[-1].clone()
    #         if (train_indices[:,t] == True).any():
    #             X_t[train_indices[:,t]] = evolve(X_t[train_indices[:,t]].mean(dim=1), cfg)[:,-1:][:,None] # [datasize, ensemble_size, 1, nx]
    #         if (train_indices[:,t] == False).any():
    #             X_t[~train_indices[:,t]] = torch.stack([split_model(model, cfg.eval_batch_size)(X_t[~train_indices[:,t], i].to(device)).cpu() for i, model in enumerate(ensemble)], dim=1) # [datasize, ensemble_size, 1, nx]
    #         preds.append(X_t)
    #     preds = torch.cat(preds, dim=2).mean(dim=1) # [datasize, nt, nx]
    # else:
    #     raise ValueError(f'Invalid mode {mode}')
    
    # First, just use ensemble to predict trajectory
    #############################
    # preds = [torch_expand(X[:,None], 1, len(ensemble))] # [datasize, ensemble_size, 1, nx]
    # for t in range(nt-1):
    #     X_t = preds[-1].clone()
    #     X_t = torch.stack([split_model(model, cfg.eval_batch_size)(X_t[:, i].to(device)).cpu() for i, model in enumerate(ensemble)], dim=1) # [datasize, ensemble_size, 1, nx] 
    #     preds.append(X_t)
    # mean = torch.cat(preds, dim=2).mean(dim=1) # [datasize, nt, nx]
    # variance = torch.cat(preds, dim=2).var(dim=1) # [datasize, nt, nx]
    # variance_per_data = variance.mean(dim=(1,2)) # [datasize]
    # norm_per_data = torch.norm(mean, dim=(1,2)) # [datasize]
    # rel = variance_per_data / norm_per_data # [datasize]
    # ready = rel < threshold # [datasize]
    # print(f'Ready: {ready.sum().item()} out of {datasize}')

    # # push all non-ready data to the beginning
    # train_indices[~ready] = torch.arange(nt-1)[None,:] < train_indices[~ready].sum(dim=1)[:,None]
    #############################
    ready = torch.ones(datasize).bool()

    try:
        # for scale_threhold in [2, 1.5, 1.25, 1.1]:
        preds = [torch_expand(X[:,None], 1, len(ensemble))] # [datasize, ensemble_size, 1, nx]
        for t in range(nt-1):
            X_t = preds[-1].clone()
            #TODO: use scale_threshold to predict simulation instability
            if (train_indices[:,t] == True).any():
                X_t[train_indices[:,t]] = evolve(X_t[train_indices[:,t], :].mean(dim=1), cfg)[:,-1:][:,None] # [datasize, ensemble_size, 1, nx]
            if (train_indices[:,t] == False).any():
                X_t[~train_indices[:,t]] = torch.stack([split_model(model, cfg.eval_batch_size)(X_t[~train_indices[:,t], i].to(device)).cpu() for i, model in enumerate(ensemble)], dim=1) # [datasize, ensemble_size, 1, nx]

            preds.append(X_t)
        preds = torch.cat(preds, dim=2).mean(dim=1) # [datasize, nt, nx]
    except:
        raise ValueError('Simulation instability')
    
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
    
    for datasize in prelim_datasize:
        print(f'Running for datasize {datasize}...')
        ensemble, metrics = create_ensemble(datasize, cfg) # list of models
        os.makedirs(path + f'{datasize}', exist_ok=True)
        if cfg.prelim_save:
            torch.save(ensemble, path + f'{datasize}/ensemble.pt')
            torch.save(metrics, path + f'{datasize}/metrics.pt')
        print(metrics)
        for p in p_list:
            print(f'Creating synthetic data for p={p}...')
            synthetic_data = create_synthetic_data(ensemble, p, cfg)
            if cfg.prelim_save:
                torch.save(synthetic_data, path + f'{datasize}/synthetic_data_p{p}.pt')

if __name__ == '__main__':
    main()

