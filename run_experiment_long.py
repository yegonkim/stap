import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm


from eval_utils import compute_metrics
from utils import set_seed, flatten_configdict, trajectory_model

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
    num_acquire = cfg.num_acquire
    device = cfg.device
    epochs = cfg.train.epochs
    lr = cfg.train.lr
    batch_size = cfg.train.batch_size
    initial_datasize = cfg.initial_datasize
    whole_initial_datasize = cfg.whole_initial_datasize
    p = cfg.p

    def train(Y, train_indices, **kwargs):
        assert unrolling == 0

        acquire_step = kwargs.get('acquire_step', 0)

        model = FNO(n_modes=cfg.model.n_modes, hidden_channels=64,
                    in_channels=1, out_channels=1)

        model = model.to(device)

        if train_indices.sum().item() == 0:
            return model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        # inputs = []
        # outputs = []
        # for b in range(Y.shape[0]):
        #     for t in range(train_nts[b].item()-1):
        #         inputs.append(Y[b,t])
        #         outputs.append(Y[b, t+1])
        # inputs = torch.stack(inputs, dim=0).unsqueeze(1)
        # outputs = torch.stack(outputs, dim=0).unsqueeze(1)

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

    def test_trajectory(model):
        X_test = Traj_dataset.traj_test[:,0:(nt-1)*timestep:timestep] # [datasize, nt-1, nx]
        Y_test = Traj_dataset.traj_test[:,timestep:nt*timestep:timestep] # [datasize, nt-1, nx]
        X_test = X_test.flatten(0, 1).unsqueeze(1) # [datasize*(nt-1), 1, nx]
        Y_test = Y_test.flatten(0, 1).unsqueeze(1) # [datasize*(nt-1), 1, nx]
        # X_test = Traj_dataset.traj_test[:,0].unsqueeze(1).to(device)
        # Y_test = Traj_dataset.traj_test[:,timestep::timestep].to(device)

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


    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    # assert timestep == 10 # hardcoded for now (130/ (14-1) = 10)
    print(f'Timestep: {timestep}')

    X = Traj_dataset.traj_train[:,0].unsqueeze(1)
    Y = Traj_dataset.traj_train[:,0::timestep]

    train_indices = torch.zeros(X.shape[0], nt-1, dtype=torch.bool)
    train_indices[:whole_initial_datasize, :] = True
    train_indices[whole_initial_datasize:whole_initial_datasize+initial_datasize, :] = torch.bernoulli(torch.ones(initial_datasize, nt-1) * p).bool()
    # values are between 1 and 14, inclusive
    # 1 means only initial data, 14 means all data

    ensemble = [train(Y, train_indices, acquire_step=0) for _ in tqdm(range(ensemble_size))]

    results = {'datasize': [], 'l2': [], 'mse': [], 'rel_l2': [], 'l2_trajectory': [], 'mse_trajectory': [], 'rel_l2_trajectory': []}

    results['datasize'].append(train_indices.sum().item())
    # rel_l2_list = [test(direct_model(model, nt-1))[1].mean().item() for model in ensemble]
    metrics_list = torch.stack([torch.stack(test_per_trajectory(trajectory_model(model, nt-1))) for model in ensemble]) # [ensemble_size, 3, datasize]
    results['l2_trajectory'].append(metrics_list[:, 0, :].mean().item())
    results['rel_l2_trajectory'].append(metrics_list[:, 1, :].mean().item())
    results['mse_trajectory'].append(metrics_list[:, 2, :].mean().item())
    metrics_list = torch.stack([torch.stack(test_trajectory(model)) for model in ensemble]) # [ensemble_size, 3, datasize]
    results['l2'].append(metrics_list[:, 0, :].mean().item())
    results['rel_l2'].append(metrics_list[:, 1, :].mean().item())
    results['mse'].append(metrics_list[:, 2, :].mean().item())
    print(f'Datasize: {results["datasize"][-1]}, L2: {results["l2"][-1]}, Rel_l2: {results["rel_l2"][-1]}, MSE: {results["mse"][-1]}, L2_trajectory: {results["l2_trajectory"][-1]}, Rel_l2_trajectory: {results["rel_l2_trajectory"][-1]}, MSE_trajectory: {results["mse_trajectory"][-1]}')
    wandb.log({'datasize': results['datasize'][-1], f'l2': results['l2'][-1], f'rel_l2': results['rel_l2'][-1], f'mse': results['mse'][-1], f'l2_trajectory': results['l2_trajectory'][-1], f'rel_l2_trajectory': results['rel_l2_trajectory'][-1], f'mse_trajectory': results['mse_trajectory'][-1]})
    
    for acquire_step in range(1, num_acquire+1):
        train_indices[whole_initial_datasize:whole_initial_datasize+initial_datasize*(2**acquire_step)] = torch.bernoulli(torch.ones(initial_datasize*(2**acquire_step), nt-1) * p).bool()
        ensemble = [train(Y, train_indices, acquire_step=acquire_step) for _ in tqdm(range(ensemble_size))]

        results['datasize'].append(train_indices.sum().item())
        # rel_l2_list = [test(direct_model(model, nt-1))[1].mean().item() for model in ensemble]
        metrics_list = torch.stack([torch.stack(test_per_trajectory(trajectory_model(model, nt-1))) for model in ensemble]) # [ensemble_size, 3, datasize]
        results['l2_trajectory'].append(metrics_list[:, 0, :].mean().item())
        results['rel_l2_trajectory'].append(metrics_list[:, 1, :].mean().item())
        results['mse_trajectory'].append(metrics_list[:, 2, :].mean().item())
        metrics_list = torch.stack([torch.stack(test_trajectory(model)) for model in ensemble]) # [ensemble_size, 3, datasize]
        results['l2'].append(metrics_list[:, 0, :].mean().item())
        results['rel_l2'].append(metrics_list[:, 1, :].mean().item())
        results['mse'].append(metrics_list[:, 2, :].mean().item())
        print(f'Datasize: {results["datasize"][-1]}, L2: {results["l2"][-1]}, Rel_l2: {results["rel_l2"][-1]}, MSE: {results["mse"][-1]}, L2_trajectory: {results["l2_trajectory"][-1]}, Rel_l2_trajectory: {results["rel_l2_trajectory"][-1]}, MSE_trajectory: {results["mse_trajectory"][-1]}')
        wandb.log({'datasize': results['datasize'][-1], f'l2': results['l2'][-1], f'rel_l2': results['rel_l2'][-1], f'mse': results['mse'][-1], f'l2_trajectory': results['l2_trajectory'][-1], f'rel_l2_trajectory': results['rel_l2_trajectory'][-1], f'mse_trajectory': results['mse_trajectory'][-1]})
    return results


def mean_std_normalize():
    assert Traj_dataset.traj_train is not None
    mean = Traj_dataset.traj_train[:32].mean()
    std = Traj_dataset.traj_train[:32].std()
    print(f'Mean: {mean}, Std: {std}')
    Traj_dataset.traj_train = (Traj_dataset.traj_train - mean) / std
    if Traj_dataset.traj_valid is not None:
        Traj_dataset.traj_valid = (Traj_dataset.traj_valid - mean) / std
    Traj_dataset.traj_test = (Traj_dataset.traj_test - mean) / std
    Traj_dataset.mean = mean
    Traj_dataset.std = std

def max_min_normalize():
    assert Traj_dataset.traj_train is not None
    max_val = Traj_dataset.traj_train[:32].max()
    min_val = Traj_dataset.traj_train[:32].min()
    mean = (max_val + min_val) / 2
    std = (max_val - min_val) / 2
    print(f'Max: {max_val}, Min: {min_val}')
    Traj_dataset.traj_train = (Traj_dataset.traj_train - mean) / std
    if Traj_dataset.traj_valid is not None:
        Traj_dataset.traj_valid = (Traj_dataset.traj_valid - mean) / std
    Traj_dataset.traj_test = (Traj_dataset.traj_test - mean) / std
    Traj_dataset.mean = mean
    Traj_dataset.std = std

@hydra.main(version_base=None, config_path="cfg_long", config_name="config.yaml")
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
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:cfg.datasize, :131], dtype=torch.float32)
    print('Loading test data...')
    with h5py.File(cfg.dataset.test_path, 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:cfg.testsize, :131], dtype=torch.float32)

    if cfg.equation == 'Heat' or cfg.equation == 'KS':
        max_min_normalize()
    else:
        mean_std_normalize()
    run_experiment(cfg)

    wandb.finish()

if __name__ == '__main__':
    main()

