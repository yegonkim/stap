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
from utils import set_seed


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with h5py.File('data/KdV_train_1024_default.h5', 'r') as f:
    traj_train = torch.tensor(f['train']['pde_140-256'][:], dtype=torch.float32)
with h5py.File('data/KdV_valid_1024_default.h5', 'r') as f:
    traj_valid = torch.tensor(f['valid']['pde_140-256'][:], dtype=torch.float32)
with h5py.File('data/KdV_test_4096_default.h5', 'r') as f:
    traj_test = torch.tensor(f['test']['pde_140-256'][:], dtype=torch.float32)


# Truncate trajectories 0 ~ 139 to 0 ~ 130
traj_train = traj_train[:, :131]
traj_valid = traj_valid[:, :131]
traj_test = traj_test[:, :131]

epochs = 1
lr = 0.001
batch_size = 32

def experiment_direct(datasize=1024, device='cpu'):
    models = []
    

    model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

    model = model.to(device)

    model.train()

    X_train = traj_train[:datasize,0].unsqueeze(1)
    Y_train = traj_train[:datasize,-1].unsqueeze(1)

    X_test = traj_test[:,0,:].unsqueeze(1)
    Y_test = traj_test[:,-1,:].unsqueeze(1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    testset = torch.utils.data.TensorDataset(X_test, Y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()


    model.eval()
    
    Y_test_pred = []
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            Y_test_pred.append(y_pred)
        Y_test_pred = torch.cat(Y_test_pred, dim=0).to(Y_test.device)
    
    metrics = compute_metrics(Y_test, Y_test_pred, d=1)

    return metrics


EXPERIMENT_FUNCTION = {'direct': experiment_direct, 'multi': experiment_multi, 'ar_0': experiment_ar, 'ar_1': experiment_ar}
CFG_DICT = {'direct': {}, 'multi': {'nt': 14}, 'ar_0': {'unrolling': 0, 'nt': 14}, 'ar_1': {'unrolling': 1, 'nt': 14}}

def run_experiment(experiment):
    results = {}
    results['experiment_name'] = experiment
    print('===== Experiment 1 - Direct prediction =====')
    for seed in tqdm(range(5)):
        print(f'Seed {seed}')
        set_seed(seed)
        datasize_list = [int(datasize) for datasize in 2 ** np.linspace(5,10,6)]
        rel_l2_list = []
        for datasize in datasize_list:
            metrics = EXPERIMENT_FUNCTION[experiment](datasize=datasize, device=device, **CFG_DICT[experiment])
            rel_l2_list.append(metrics[1].mean().item())
        results[seed] = {'datasize': datasize_list, 'rel_l2': rel_l2_list}
        print(f'Results: {results[seed]}')
    
    save_path = get_results_path() + f'/results_{experiment}_{time.strftime("%Y%m%d-%H%M%S")}.pt'
    torch.save(results, save_path)
    print(f'Results saved to {save_path}')

def main():
    parser = argparse.ArgumentParser(description="Run experiment of your choice")
    parser.add_argument("--experiment", choices=["direct", "multi", "ar_0", "ar_1"], default="direct")

    args = parser.parse_args()

    run_experiment(args.experiment)

if __name__ == '__main__':
    main()