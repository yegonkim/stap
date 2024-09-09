import torch
import numpy as np

from . import ACQUISITION_FUNCTIONS
from .feature_functions import get_features_ycov
from utils import torch_delete

from tqdm import tqdm


def select(ensemble, X_train, X_pool, batch_acquire, selection_method='random', **cfg):
    device = cfg.get('device', 'cpu')
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
        
        if "invidual" in selection_method.split("_"):
            with torch.no_grad():
                features_all = get_features_ycov(X_all, ensemble) # [bs_all, N, dim]
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
                        distances = torch.cdist(features_all[sel_indices][None], features_all[rem_indices][None]).squeeze(0).square() # [bs_sel, bs_rem]
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
                        d = d.reshape(-1, features_all.shape[1]).mean(dim=0) # [-1, N]
                        d = d.sum(dim=1) # [-1]
                        new = torch.argmax(d)

                sel = torch.cat([sel, rem[new:new+1]], dim=0)
                new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
                rem = torch_delete(rem, new)
                
        elif "concat" in selection_method.split("_"):
            with torch.no_grad():
                features_all = get_features_ycov(X_all, ensemble) # [bs_all, feature_dim]
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
        new_idxs = new_sel
    # elif selection_method == "lcmd_shared":
    #     pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
    #     new_idxs = torch.tensor([], dtype=torch.int64, device=device)

    #     X_all = torch.cat([X_pool, X_train], dim=0)
    #     rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
    #     sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
    #     new_sel = torch.tensor([], dtype=torch.int64, device=device)
        
    #     with torch.no_grad():
    #         features_all = get_features_ycov(X_all, ensemble) # [bs_all, N, dim]
    #         indices_all = torch.arange(X_all.shape[0], device=device, dtype=torch.int64).unsqueeze(1).repeat(1, features_all.shape[1]) # [bs_all, N]
    #         features_all = features_all.flatten(start_dim=0, end_dim=1).flatten(start_dim=1) # [bs_all*N, dim]
    #         indices_all = indices_all.flatten(start_dim=0) # [bs_all*N]

    #     for _ in range(batch_acquire):
    #         rem_indices = torch.nonzero(torch.isin(indices_all, rem)).squeeze(1) # [bs_rem]
    #         sel_indices = torch.nonzero(torch.isin(indices_all, sel)).squeeze(1) # [bs_sel]
    #         if sel.shape[0] == 0:
    #             new = torch.argmax(torch.norm(features_all[rem_indices], dim=1))
    #             new = indices_all[rem_indices][new]
    #         else:
    #             distances = torch.cdist(features_all[sel_indices][None], features_all[rem_indices][None]).squeeze(0).square() # [bs_sel, bs_rem]
    #             centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
    #             largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel_indices))])) # index in bs_sel
    #             mask = torch.where(centers == largest_cluster, 0.0, -1e10)
    #             new = torch.argmax(distances[largest_cluster] + mask) # index in rem
    #             new = indices_all[rem_indices][new]
    #             # new = indices_all[new]
    #         # find new in rem
    #         new = torch.nonzero(rem == new).item()
    #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
    #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
    #         rem = torch_delete(rem, new)
    #     new_idxs = new_sel
    # elif selection_method == "lcmd_p":
    #     pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
    #     new_idxs = torch.tensor([], dtype=torch.int64, device=device)


        
    #     with torch.no_grad():
    #         features_all = get_features_ycov(X_all, ensemble) # [bs_all, feature_dim]
    #         features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]
        
    #     for _ in range(batch_acquire):
    #         if sel.shape[0] == 0:
    #             new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
    #         else:
    #             distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
    #             centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
    #             largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel))])) # index in bs_sel
    #             mask = torch.where(centers == largest_cluster, 0.0, -1e10)
    #             new = torch.argmax(distances[largest_cluster] + mask) # index in rem
    #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
    #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
    #         rem = torch_delete(rem, new)
    #     new_idxs = new_sel
    # elif selection_method == "lcmd_p_shared":
    #     pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
    #     new_idxs = torch.tensor([], dtype=torch.int64, device=device)

    #     X_all = X_pool
    #     rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
    #     sel = torch.tensor([], dtype=torch.int64, device=device)
    #     new_sel = torch.tensor([], dtype=torch.int64, device=device)
        
    #     with torch.no_grad():
    #         features_all = get_features_ycov(X_all, ensemble) # [bs_all, N, dim]
    #         indices_all = torch.arange(X_all.shape[0], device=device, dtype=torch.int64).unsqueeze(1).repeat(1, features_all.shape[1]) # [bs_all, N]
    #         features_all = features_all.flatten(start_dim=0, end_dim=1).flatten(start_dim=1) # [bs_all*N, dim]
    #         indices_all = indices_all.flatten(start_dim=0) # [bs_all*N]

    #     for _ in range(batch_acquire):
    #         rem_indices = torch.nonzero(torch.isin(indices_all, rem)).squeeze(1) # [bs_rem]
    #         sel_indices = torch.nonzero(torch.isin(indices_all, sel)).squeeze(1) # [bs_sel]
    #         if sel.shape[0] == 0:
    #             new = torch.argmax(torch.norm(features_all[rem_indices], dim=1))
    #             new = indices_all[rem_indices][new]
    #         else:
    #             distances = torch.cdist(features_all[sel_indices][None], features_all[rem_indices][None]).squeeze(0).square() # [bs_sel, bs_rem]
    #             centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
    #             largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel_indices))])) # index in bs_sel
    #             mask = torch.where(centers == largest_cluster, 0.0, -1e10)
    #             new = torch.argmax(distances[largest_cluster] + mask) # index in rem
    #             new = indices_all[rem_indices][new]
    #             # new = indices_all[new]
    #         # find new in rem
    #         new = torch.nonzero(rem == new).item()
    #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
    #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
    #         rem = torch_delete(rem, new)
    #     new_idxs = new_sel
    else:
        raise ValueError(f"Selection method {selection_method} not implemented.")

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
            elif selection_method == "lcmd":
                
                pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
                new_idxs = torch.tensor([], dtype=torch.int64, device=device)

                X_all = torch.cat([X_pool, X_train], dim=0)
                rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
                sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
                new_sel = torch.tensor([], dtype=torch.int64, device=device)
                
                # if cfg.acquisition.mode.split("_")[0] == "p":
                #     X_all = X_pool
                #     rem = torch.arange(X_all.shape[0], device=device, dtype=torch.int64)
                #     sel = torch.tensor([], dtype=torch.int64, device=device)
                #     new_sel = torch.tensor([], dtype=torch.int64, device=device)
                # elif cfg.acquisition.mode.split("_")[0] == "tp":
                #     X_all = torch.cat([X_pool, self.train_list['X']], dim=0)
                #     rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
                #     sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
                #     new_sel = torch.tensor([], dtype=torch.int64, device=device)

                with torch.no_grad():
                    features_all = get_features_ycov(X_all, ensemble) # [bs_all, feature_dim]
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
                new_idxs = new_sel
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
