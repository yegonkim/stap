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
from acquisition.acquirers import select


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Traj_dataset:
    traj_train = None
    traj_valid = None
    traj_test = None


# epochs = 1
# lr = 0.001
# batch_size = 32
# batch_acquire = 1

epochs = 500
lr = 0.001
batch_size = 32
batch_acquire = 32


def experiment_al_comb(initial_datasize=256, batch_acquire=32, num_acquire=1, device='cpu', **cfg):
    unrolling = cfg.get('unrolling', 1)
    nt = cfg.get('nt', 14)
    ensemble_size = cfg.get('ensemble_size', 5)
    selection_method = cfg.get('selection_method', 'random')
    acquisition_function = cfg.get('acquisition_function', 'variance')
    features = cfg.get('feature', 'direct')

    def train(X_train, Y_train):
        model = FNO(n_modes=(256, ), hidden_channels=64,
                    in_channels=1, out_channels=1)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
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
            Y_test_pred = torch.cat(Y_test_pred, dim=0).to(Y_test.device)
        
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
            Y_test_pred = torch.cat(Y_test_pred, dim=0).to(Y_test.device)
        
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
            return torch.cat(trajectory, dim=1) # [batch_size, unrolling, nx]

    class direct_combination_model(torch.nn.Module):
        def __init__(self, models, unrolling):
            super().__init__()
            self.models = models
            self.unrolling = unrolling
            self.num_models = len(models)
            self.combination = torch.randint(0, self.num_models, (self.unrolling,))
        def forward(self, x):
            for _ in range(self.unrolling):
                x = self.models[self.combination[_]](x)
            return x

    class trajectory_combination_model(torch.nn.Module):
        def __init__(self, models, unrolling):
            super().__init__()
            self.models = models
            self.unrolling = unrolling
            self.num_models = len(models)
            self.combination = torch.randint(0, self.num_models, (self.unrolling,))
        def forward(self, x):
            trajectory = []
            for _ in range(self.unrolling):
                x = self.models[self.combination[_]](x)
                trajectory.append(x)
            return torch.cat(trajectory, dim=1)

    timestep = (Traj_dataset.traj_train.shape[1] - 1) // (nt - 1) # 10
    assert timestep == 10

    X = Traj_dataset.traj_train[:,0].unsqueeze(1).to(device)
    Y = Traj_dataset.traj_train[:,0::timestep].to(device)

    train_idxs = torch.arange(initial_datasize, device=device)
    pool_idxs = torch.arange(initial_datasize, X.shape[0], device=device)

    X_train = X[train_idxs]
    Y_train = Y[train_idxs]

    X_pool = X[pool_idxs]

    ensemble = [train(X_train, Y_train) for _ in tqdm(range(ensemble_size))]

    results = {'datasize': [], 'rel_l2': [], 'rel_l2_trajectory': []}


    results['datasize'].append(train_idxs.shape[0])
    rel_l2_list = [test(direct_model(model, nt-1))[1].mean().item() for model in ensemble]
    rel_l2_trajectory_list = [test_trajectory(trajectory_model(model, nt-1))[1].mean().item() for model in ensemble]
    results['rel_l2'].append(torch.mean(torch.tensor(rel_l2_list)).item())
    results['rel_l2_trajectory'].append(torch.mean(torch.tensor(rel_l2_trajectory_list)).item())
    print(f'Datasize: {results["datasize"][-1]}, Rel_l2: {results["rel_l2"][-1]}, Rel_l2_trajectory: {results["rel_l2_trajectory"][-1]}')


    for i in range(num_acquire):
        if features == 'direct':
            # unrolled_ensemble = [direct_model(model, nt-1) for model in ensemble]
            unrolled_ensemble = [direct_combination_model(ensemble, nt-1) for _ in range(100)]
        elif features == 'trajectory':
            unrolled_ensemble = [trajectory_combination_model(ensemble, nt-1) for _ in range(100)]
        new_idxs = select(unrolled_ensemble, X_train, X_pool, batch_acquire, selection_method=selection_method, acquisition_function=acquisition_function, device=device)

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

        ensemble = [train(X_train, Y_train) for _ in tqdm(range(ensemble_size))]

        results['datasize'].append(train_idxs.shape[0])
        rel_l2_list = [test(direct_model(model, nt-1))[1].mean().item() for model in ensemble]
        rel_l2_trajectory_list = [test_trajectory(trajectory_model(model, nt-1))[1].mean().item() for model in ensemble]
        results['rel_l2'].append(torch.mean(torch.tensor(rel_l2_list)).item())
        results['rel_l2_trajectory'].append(torch.mean(torch.tensor(rel_l2_trajectory_list)).item())
        print(f'Datasize: {results["datasize"][-1]}, Rel_l2: {results["rel_l2"][-1]}, Rel_l2_trajectory: {results["rel_l2_trajectory"][-1]}')
    
    return results

# EXPERIMENT_FUNCTION = {'direct': experiment_direct, 'multi': experiment_multi, 'ar_0': experiment_ar, 'ar_1': experiment_ar}
CFG_DICT = {'direct_random': {'nt': 14, 'ensemble_size': 5, 'selection_method': 'random', 'acquisition_function': 'variance', 'feature': 'direct'},
            'direct_variance': {'nt': 14, 'ensemble_size': 5, 'selection_method': 'greedy', 'acquisition_function': 'variance', 'feature': 'direct'},
            'direct_lcmd': {'nt': 14, 'ensemble_size': 5, 'selection_method': 'lcmd', 'acquisition_function': 'variance', 'feature': 'direct'},
            'trajectory_random': {'nt': 14, 'ensemble_size': 5, 'selection_method': 'random', 'acquisition_function': 'variance', 'feature': 'trajectory'},
            'trajectory_variance': {'nt': 14, 'ensemble_size': 5, 'selection_method': 'greedy', 'acquisition_function': 'variance', 'feature': 'trajectory'},
            'trajectory_lcmd': {'nt': 14, 'ensemble_size': 5, 'selection_method': 'lcmd', 'acquisition_function': 'variance', 'feature': 'trajectory'}}

def run_experiment(experiment, equation, **cfg):
    results = {}
    results['experiment_name'] = experiment
    results['equation_name'] = equation
    cfg.update(CFG_DICT[experiment])

    datasize_list = [int(datasize) for datasize in 2 ** np.linspace(5,9,5)]
    # datasize_list = [20,30]
    results['initial_datasize_list'] = datasize_list

    for seed in range(5):
        print(f'Seed {seed}')
        set_seed(seed)
        results[seed] = {}
        
        for initial_datasize in datasize_list:
            results_instance = experiment_al_comb(initial_datasize=initial_datasize, batch_acquire=batch_acquire, num_acquire=1, device=device, **cfg)
            results[seed][initial_datasize] = results_instance
        print(results[seed])

    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiment of your choice")
    parser.add_argument("--equation", choices=["KdV", "Burgers", "KS"], default="KdV")
    parser.add_argument("--experiment", choices=["direct_random", "direct_variance", "direct_lcmd", "trajectory_random", "trajectory_variance", "trajectory_lcmd"], default="direct_random")
    parser.add_argument("--unrolling", type=int, default=1)

    args = parser.parse_args()

    with h5py.File(f'data/{args.equation}_train_1024_default.h5', 'r') as f:
        Traj_dataset.traj_train = torch.tensor(f['train']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    with h5py.File(f'data/{args.equation}_valid_1024_default.h5', 'r') as f:
        Traj_dataset.traj_valid = torch.tensor(f['valid']['pde_140-256'][:], dtype=torch.float32)[:, :131]
    with h5py.File(f'data/{args.equation}_test_4096_default.h5', 'r') as f:
        Traj_dataset.traj_test = torch.tensor(f['test']['pde_140-256'][:], dtype=torch.float32)[:, :131]

    results = run_experiment(args.experiment, args.equation, unrolling=args.unrolling)

    print(results)
    save_path = get_results_path() + f'/results_al_comb_{args.equation}_{args.experiment}_{time.strftime("%Y%m%d-%H%M%S")}.pt'
    torch.save(results, save_path)
    print(f'Results saved to {save_path}')

if __name__ == '__main__':
    main()