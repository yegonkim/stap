import torch
import numpy as np

from . import ACQUISITION_FUNCTIONS
from .feature_functions import get_features_ycov, get_features_hidden
from utils import torch_delete, torch_expand
from .acquisition_functions import entropy

import time
from tqdm import tqdm


def select(ensemble, X_train, X_pool, batch_acquire, selection_method='random', **cfg):
    device = cfg.get('device', 'cpu')
    num_random_pool = cfg.get('num_random_pool', X_pool.shape[0])

    random_pool_idxs = torch.randperm(X_pool.shape[0])[:num_random_pool]
    X_pool = X_pool[random_pool_idxs]

    if selection_method == "random":
        weights = torch.ones(1, X_pool.shape[0])
        new_idxs = torch.multinomial(weights, num_samples=batch_acquire, replacement=False)[0]
    elif selection_method == "variance":
        X = X_pool
        with torch.no_grad():
            features = get_features_ycov(X, ensemble) # [bs, N, dim]
            trace = torch.einsum('bnd,bnd->b', features, features) # [bs]
            score = trace
        new_idxs = torch.argsort(score, descending=True)[:batch_acquire]
    elif selection_method == "stochastic":
        X = X_pool
        with torch.no_grad():
            features = get_features_ycov(X, ensemble)
            trace = torch.einsum('bnd,bnd->b', features, features)
        weights = trace / trace.sum()
        new_idxs = torch.multinomial(weights, num_samples=batch_acquire, replacement=False)
    elif "lcmd" in selection_method.split("_"):
        pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
        new_idxs = torch.tensor([], dtype=torch.int64, device=device)

        if "tp" in selection_method.split("_"):
            X_all = torch.cat([X_pool, X_train], dim=0)
            rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
            sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
            new_sel = torch.tensor([], dtype=torch.int64, device=device)
        elif "p" in selection_method.split("_"):
            X_all = X_pool
            rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
            sel = torch.tensor([], dtype=torch.int64, device=device)
            new_sel = torch.tensor([], dtype=torch.int64, device=device)
        else:
            raise ValueError(f"Selection method {selection_method} not implemented. Choose between tp and p.")
        
        if "hidden" in selection_method.split("_"):
            get_features = get_features_hidden
        elif "ycov" in selection_method.split("_"):
            get_features = get_features_ycov
        else:
            raise ValueError(f"Selection method {selection_method} not implemented. Choose between hidden and ycov.")

        with torch.no_grad():
            if "individual" in selection_method.split("_"):
                features_all = get_features(X_all, ensemble) # [bs_all, N, dim]
                N = features_all.shape[1]
                indices_all = torch.arange(X_all.shape[0], device=device, dtype=torch.int64).unsqueeze(1).repeat(1, features_all.shape[1]) # [bs_all, N]
                features_all = features_all.flatten(start_dim=0, end_dim=1).flatten(start_dim=1) # [bs_all*N, dim]
                indices_all = indices_all.flatten(start_dim=0) # [bs_all*N] e.g. [0, 0, 0, ..., 1, 1, 1, ...]
                for _ in range(batch_acquire):
                    rem_indices = torch.nonzero(torch.isin(indices_all, rem)).squeeze(1) # [bs_rem] e.g. [0, 1, 2, ...]
                    sel_indices = torch.nonzero(torch.isin(indices_all, sel)).squeeze(1) # [bs_sel]
                    if "max" in selection_method.split("_"):
                        if sel.shape[0] == 0:
                            new = torch.argmax(torch.norm(features_all[rem_indices], dim=1))
                            new = indices_all[rem_indices][new]
                            new = torch.nonzero(rem == new).item()
                        else:
                            # print('Starting distance calculation')
                            # time_start = time.time()
                            distances = torch.cdist(features_all[sel_indices][None].cpu(), features_all[rem_indices][None].cpu()).squeeze(0).square() # [bs_sel, bs_rem]
                            # print('Finished distance calculation')
                            # print(f'Time taken: {time.time()-time_start}')
                            centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
                            largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel_indices))])) # index in bs_sel
                            mask = torch.where(centers == largest_cluster, 0.0, -1e10)
                            new = torch.argmax(distances[largest_cluster] + mask) # index in rem
                            new = indices_all[rem_indices][new]
                            # new = indices_all[new]
                            new = torch.nonzero(rem == new).item()
                    elif "mean" in selection_method.split("_"):
                        if sel.shape[0] == 0:
                            new = torch.argmax(torch.norm(features_all[rem_indices], dim=1))
                            new = indices_all[rem_indices][new]
                            new = torch.nonzero(rem == new).item()
                        else:
                            distances = torch.cdist(features_all[sel_indices][None], features_all[rem_indices][None]).squeeze(0).square()
                            centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
                            largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel_indices))])) # index in bs_sel
                            # mask = torch.where(centers == largest_cluster, 0.0, -1e10)
                            mask = torch.where(centers == largest_cluster, 1, 0)
                            d = distances[largest_cluster] * mask # [bs_rem]
                            d = d.reshape(-1, N).sum(dim=0) # [-1, N]
                            new = torch.argmax(d)

                    sel = torch.cat([sel, rem[new:new+1]], dim=0)
                    new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
                    rem = torch_delete(rem, new)
                    
            elif "concat" in selection_method.split("_"):
                features_all = get_features(X_all, ensemble) # [bs_all, feature_dim]
                features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]
                for _ in range(batch_acquire):
                    if sel.shape[0] == 0:
                        new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
                    else:
                        distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
                        centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
                        largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel))])) # index in bs_sel
                        mask = torch.where(centers == largest_cluster, 0.0, -1e10)
                        new = torch.argmax(distances[largest_cluster] + mask) # index in rem
                    sel = torch.cat([sel, rem[new:new+1]], dim=0)
                    new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
                    rem = torch_delete(rem, new)
            else:
                raise ValueError(f"Selection method {selection_method} not implemented.")  
            new_idxs = new_sel
    elif "mutual" in selection_method.split("_"):
        X_target = X_pool[torch.randperm(X_pool.shape[0])[:1000]] # [20, ...]
        if "individual" in selection_method.split("_"):
            with torch.no_grad():
                entropies = entropy(X_pool[:,None], ensemble) # [bs_pool]
                joint_entropies = [entropy(torch.cat([X_pool[:,None], torch_expand(X_target[None, i:i+1], 0, X_pool.shape[0])], dim=1), ensemble) for i in range(X_target.shape[0])]
                joint_entropies = torch.stack(joint_entropies, dim=1).mean(dim=1) # [bs_pool]
                scores = entropies - joint_entropies # [bs_pool]
            if "max" in selection_method.split("_"):
                new_idxs = torch.argsort(scores, descending=True)[:batch_acquire] # [bs_acquire]
            elif "stochastic" in selection_method.split("_"):
                # Gumbel max trick
                u = torch.rand(scores.shape[0])
                gumbel_noise = -torch.log(-torch.log(u))
                scores = scores + gumbel_noise.to(scores.device)
                new_idxs = torch.argsort(scores, descending=True)[:batch_acquire] # [bs_acquire]
        elif "concat" in selection_method.split("_"):
            pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
            new_idxs = torch.tensor([], dtype=torch.int64, device=device)
            for _ in range(batch_acquire):
                current_pool_idxs = torch_delete(pool_idxs, new_idxs)
                X_pool_temp = X_pool[current_pool_idxs]
                if new_idxs.shape[0] == 0:
                    entropies = entropy(X_pool_temp[:,None], ensemble) # [bs_pool]
                    joint_entropies = [entropy(torch.cat([X_pool[:,None], torch_expand(X_target[None, i:i+1], 0, X_pool.shape[0])], dim=1), ensemble) for i in range(X_target.shape[0])]
                    joint_entropies = torch.stack(joint_entropies, dim=1).mean(dim=1) # [bs_pool]
                    scores = entropies - joint_entropies # [bs_pool]
                    new = torch.argmax(scores)
                else:
                    X = X_pool[new_idxs]
                    entropies = entropy(torch.cat([X_pool_temp[:,None], torch_expand(X[None,:],0,X_pool_temp.shape[0])], dim=1), ensemble) # [bs_pool]
                    joint_entropies = [entropy(torch.cat([X_pool_temp[:,None], torch_expand(X[None,:],0,X_pool_temp.shape[0]), torch_expand(X_target[None, i:i+1], 0, X_pool_temp.shape[0])], dim=1), ensemble) for i in range(X_target.shape[0])]
                    joint_entropies = torch.stack(joint_entropies, dim=1).mean(dim=1) # [bs_pool]
                    scores = entropies - joint_entropies
                    new = torch.argmax(scores)
                new_idxs = torch.cat([new_idxs, current_pool_idxs[new:new+1]], dim=0)           
    else:
        raise ValueError(f"Selection method {selection_method} not implemented.")

    new_idxs = random_pool_idxs[new_idxs.cpu()]
    return new_idxs

def select_time(ensemble, Y, train_nts, time_steps_acquire, selection_method='random', mode='time', **cfg):
    device = cfg.get('device', 'cpu')
    train_nts = train_nts.clone()

    bs = Y.shape[0]
    nt = Y.shape[1]

    for model in ensemble:
        model.eval()

    if not selection_method == "random":
        pool_indices = [(b, t) for b in range(bs) for t in range(train_nts[b], nt)]
        with torch.no_grad():
            features = []
            for b in tqdm(range(bs)):
                if train_nts[b] < nt:
                    trajectories = []
                    for model in ensemble:
                        trajectory = []
                        trajectory.append(Y[b, train_nts[b]-1])
                        for t in range(nt-train_nts[b]):
                            trajectory.append(model(trajectory[-1][None,None,...])[0,0])
                        trajectory.pop(0)
                        trajectory = torch.stack(trajectory, dim=0)
                        trajectories.append(trajectory)
                    trajectories = torch.stack(trajectories, dim=1) # [nt-train_nt, num_models, ...]
                    feature = trajectories - trajectories.mean(dim=1, keepdim=True) # [nt-train_nt, num_models, ...]
                    features.append(feature)
            features = torch.cat(features, dim=0) # [bs*(nt-train_nt), num_models, ...]
            features = features.flatten(start_dim=2) # [bs*(nt-train_nt), num_models, dim]
        assert features.shape[0] == len(pool_indices)
        # features_dict = {pool_indices[i]: features[i] for i in range(features.shape[0])}

        scores = torch.einsum('bnd,bnd->b', features, features) # [bs*(nt-train_nt)]
        # torch.save(scores_dict, 'scores_dict.pt')

        scores_temp = torch.zeros(bs, nt)
        for i, (b,t) in enumerate(pool_indices):
            scores_temp[b,t] = scores[i]
        scores = scores_temp

    if mode == 'time':
        for i in tqdm(range(time_steps_acquire)):
            if selection_method == "random":
                weights = (train_nts < nt).float()
                new_idx = torch.multinomial(weights, num_samples=1, replacement=False)[0]
            elif selection_method == "variance":
                scores_candidates = torch.stack([scores[b, train_nts[b].item()] for b in range(bs) if train_nts[b].item() < nt], dim=0)
                new_idx = torch.argmax(scores_candidates)
                indices = [b for b in range(bs) if train_nts[b].item() < nt]
                new_idx = indices[new_idx]
            elif selection_method == "stochastic":
                scores_candidates = torch.stack([scores[b, train_nts[b].item()] for b in range(bs) if train_nts[b].item() < nt], dim=0)
                new_idx = torch.multinomial(scores_candidates, num_samples=1, replacement=False)[0]
                indices = [b for b in range(bs) if train_nts[b].item() < nt]
                new_idx = indices[new_idx]
            else:
                raise ValueError(f"Selection method {selection_method} not implemented.")
            assert train_nts[new_idx].item() < nt
            train_nts[new_idx] += 1
    elif mode == 'whole':
        assert torch.all(torch.logical_or(train_nts == nt, train_nts == 1)) # all or nothing
        batch_acquire = time_steps_acquire // (nt-1)
        if selection_method == "random":
            weights = (train_nts < nt).float()
            new_idxs = torch.multinomial(weights, num_samples=batch_acquire, replacement=False)
        elif selection_method == "variance":
            scores_trajectories = torch.tensor([sum([scores[b, t] for t in range(1, nt)]) for b in range(bs) if train_nts[b].item() < nt], device=device)
            new_idxs = torch.argsort(scores_trajectories, descending=True)[:batch_acquire]
            indices = torch.tensor([b for b in range(bs) if train_nts[b].item() < nt], device=device)
            new_idxs = indices[new_idxs]
        elif selection_method == "stochastic":
            scores_trajectories = torch.tensor([sum([scores[b, t] for t in range(1, nt)]) for b in range(bs) if train_nts[b].item() < nt], device=device)
            new_idxs = torch.multinomial(scores_trajectories, num_samples=batch_acquire, replacement=False)
            indices = torch.tensor([b for b in range(bs) if train_nts[b].item() < nt], device=device)
            new_idxs = indices[new_idxs]
        else:
            raise ValueError(f"Selection method {selection_method} not implemented.")
    
        for i in range(batch_acquire):
            assert train_nts[new_idxs[i]].item() < nt
            train_nts[new_idxs[i]] = nt

    return train_nts

