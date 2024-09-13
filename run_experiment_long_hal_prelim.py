import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm
import os

from eval_utils import compute_metrics
from utils import set_seed, flatten_configdict, trajectory_model, ensemble_mean_model

from omegaconf import OmegaConf
import hydra

class Traj_dataset:
    traj_train = None
    traj_valid = None
    traj_test = None


def create_ensemble(datasize, cfg):
    print('Creating model...')
    device = cfg.device
    
    unrolling = cfg.train.unrolling
    nt = cfg.nt
    # ensemble_size = cfg.ensemble_size
    # fix ensemble size to 1 for now
    ensemble_size = 1
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size

    def train(Y, train_indices, **kwargs):
        model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

        model = model.to(device)

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
        results = {'l2_per_model': [], 'mse_per_model': [], 'rel_l2_per_model': []}
        metrics_list = torch.stack([torch.stack(test_per_trajectory(trajectory_model(model, nt-1))) for model in ensemble]) # [ensemble_size, 3, datasize]
        results['l2_per_model'].append(metrics_list[:, 0, :].mean().item())
        results['rel_l2_per_model'].append(metrics_list[:, 1, :].mean().item())
        results['mse_per_model'].append(metrics_list[:, 2, :].mean().item())
        metrics_list = torch.stack(test_per_trajectory(trajectory_model(ensemble_mean_model(ensemble), nt-1))) # [3, datasize]
        results['l2_timestep'].append(metrics_list[0, :].mean().item())
        results['rel_l2_timestep'].append(metrics_list[1, :].mean().item())
        results['mse_per_timestep'].append(metrics_list[2, :].mean().item())
        metrics_list = torch.stack(test_per_trajectory(ensemble_mean_model(trajectory_model(ensemble, nt-1)))) # [3, datasize]
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


def create_synthetic_data(model, cfg, mode='single'):
    print('Creating synthetic data...')
    device = cfg.device

    max_prelim_datasize = max(cfg.prelim_datasize)
    datasize = cfg.datasize - max_prelim_datasize
    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (cfg.nt - 1) # 10
    # assert timestep == 10 # hardcoded for now (130/ (14-1) = 10)
    print(f'Timestep: {timestep}')

    X = Traj_dataset.traj_train[:datasize,0:1] # [datasize, 1, nx]

    

    return Y

@hydra.main(version_base=None, config_path="cfg_selective", config_name="config.yaml")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    prelim_datasize = cfg.prelim_datasize

    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    print('Loading training data...')
    with h5py.File(cfg.dataset.train_path, 'r') as f:
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:cfg.datasize, :131], dtype=torch.float32)
    print('Loading test data...')
    with h5py.File(cfg.dataset.test_path, 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:cfg.testsize, :131], dtype=torch.float32)

    for datasize in prelim_datasize:
        print(f'Running for datasize {datasize}...')
        ensemble, metrics = create_ensemble(datasize, cfg) # list of models
        os.makedirs(f'long_hal_prelim/{datasize}', exist_ok=True)
        torch.save(ensemble, f'long_hal_prelim/{datasize}/ensemble.pt')
        torch.save(metrics, f'long_hal_prelim/{datasize}/metrics.pt')
        print(metrics)
        synthetic_data = create_synthetic_data(ensemble, cfg, mode='single')
        torch.save(synthetic_data, f'long_hal_prelim/{datasize}/synthetic_data_single.pt')
        synthetic_data = create_synthetic_data(ensemble, cfg, mode='timestep')
        torch.save(synthetic_data, f'long_hal_prelim/{datasize}/synthetic_data_timestep.pt')
        synthetic_data = create_synthetic_data(ensemble, cfg, mode='trajectory')
        torch.save(synthetic_data, f'long_hal_prelim/{datasize}/synthetic_data_trajectory.pt')

if __name__ == '__main__':
    main()

