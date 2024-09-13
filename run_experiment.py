import h5py
import torch
import numpy as np
from neuralop.models import FNO
from tqdm import tqdm
import random

import argparse
import time

from eval_utils import compute_metrics

from utils import set_seed


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Traj_dataset:
    traj_train = None
    traj_valid = None
    traj_test = None


epochs = 500
lr = 0.001
batch_size = 32

def experiment_direct(datasize=1024, device='cpu'):

    model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

    model = model.to(device)

    model.train()

    X_train = Traj_dataset.traj_train[:datasize,0].unsqueeze(1)
    Y_train = Traj_dataset.traj_train[:datasize,-1].unsqueeze(1)

    X_test = Traj_dataset.traj_test[:,0,:].unsqueeze(1)
    Y_test = Traj_dataset.traj_test[:,-1,:].unsqueeze(1)
    
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


def experiment_multi(datasize=1024, device='cpu', **cfg):
    nt = cfg.get('nt', 14)

    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    assert timestep == 10

    model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=nt)

    model = model.to(device)

    model.train()

    X_train = Traj_dataset.traj_train[:datasize,0].unsqueeze(1)
    Y_train = Traj_dataset.traj_train[:datasize,0::timestep]

    assert Y_train.shape[1] == nt

    X_test = Traj_dataset.traj_test[:,0,:].unsqueeze(1)
    Y_test = Traj_dataset.traj_test[:,-1,:].unsqueeze(1)


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
            y_pred = model(x)[:,-1:]
            Y_test_pred.append(y_pred)
        Y_test_pred = torch.cat(Y_test_pred, dim=0).to(Y_test.device)
    
    metrics = compute_metrics(Y_test, Y_test_pred, d=1)

    return metrics

def experiment_ar(datasize=1024, device='cpu', **cfg):
    unrolling = cfg.get('unrolling', 1)
    nt = cfg.get('nt', 14)

    model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

    model = model.to(device)

    model.train()

    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    assert timestep == 10

    X_train = Traj_dataset.traj_train[:datasize,0].unsqueeze(1)
    Y_train = Traj_dataset.traj_train[:datasize,0::timestep]

    assert Y_train.shape[1] == nt

    X_test = Traj_dataset.traj_test[:,0,:].unsqueeze(1)
    Y_test = Traj_dataset.traj_test[:,-1,:].unsqueeze(1)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    testset = torch.utils.data.TensorDataset(X_test, Y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    pbar = range(epochs)
    for epoch in pbar:
        model.train()
        max_unrolling = epoch if epoch <= unrolling else unrolling
        unrolling_list = [r for r in range(max_unrolling + 1)]

        # Loop over every epoch as often as the number of timesteps in one trajectory.
        # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
        # Therefore in expectation the whole available training information is covered.
        total_loss = 0
        for i in range(nt):
            for x, y in dataloader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device) # y has shape [batch_size, nt, nx]

                unrolled = random.choice(unrolling_list)
                bs = x.shape[0]

                steps = [t for t in range(0, nt - 1 - unrolled)]
                random_steps = random.choices(steps, k=bs)

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


        # if (epoch+1) % 10 == 0:    
        #     model.eval()
        
        #     Y_test_pred = []
        #     with torch.no_grad():
        #         for x, y in testloader:
        #             x, y = x.to(device), y.to(device)
        #             for _ in range(nt-1):
        #                 x = model(x)
        #             Y_test_pred.append(x)
        #         Y_test_pred = torch.cat(Y_test_pred, dim=0).to(Y_test.device)
            
        #     metrics = compute_metrics(Y_test, Y_test_pred, d=1)
        #     print(f'Epoch {epoch}: {metrics[1].mean().item()}')

    model.eval()
    
    Y_test_pred = []
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            for _ in range(nt-1):
                x = model(x)
            Y_test_pred.append(x)
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
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiment of your choice")
    parser.add_argument("--equation", choices=["KdV", "Burgers", "KS"], default="KdV")
    parser.add_argument("--experiment", choices=["direct", "multi", "ar_0", "ar_1"], default="direct")

    args = parser.parse_args()

    with h5py.File(f'data/{args.equation}_train_1024_default.h5', 'r') as f:
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    with h5py.File(f'data/{args.equation}_valid_1024_default.h5', 'r') as f:
        Traj_dataset.traj_valid = torch.tensor(f['valid']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    with h5py.File(f'data/{args.equation}_test_4096_default.h5', 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:], dtype=torch.float32)[:, :131]

    results = run_experiment(args.experiment)

    print(results)
    save_path = get_results_path() + f'/results_{args.equation}_{args.experiment}_{time.strftime("%Y%m%d-%H%M%S")}.pt'
    torch.save(results, save_path)
    print(f'Results saved to {save_path}')

if __name__ == '__main__':
    main()