import torch
from tqdm import tqdm
from .feature_functions import get_features_hidden_trajectory, get_features_ycov_trajectory
from utils import direct_model, trajectory_model, split_model, torch_delete, torch_expand
import numpy as np
from .acquisition_function import EER_Calculator

class Acquirer:
    def __init__(self, ensemble, pool, L, train_indices, cfg):
        self.device = cfg.device
        self.ensemble = ensemble
        self.pool = pool
        self.L = L
        self.train_indices = train_indices # dictionary {index: S}
        self.cfg = cfg
        self.eval_batch_size = cfg.eval_batch_size
        self.initial_selection_method = cfg.initial_selection_method
        self.post_selection_method = cfg.post_selection_method
        self.eer_mode = 'MF-EER' if cfg.mean_field else 'EER'

    def _eval_mode(self):
        for model in self.ensemble:
            model.eval()

    @torch.no_grad()
    def initialize_selection(self):
        self._eval_mode()
        pool = self.pool
        bs = len(pool)

        selection_method = self.initial_selection_method

        have_seen = torch.zeros(bs, dtype=torch.bool)
        for index in self.train_indices.keys():
            have_seen[index] = True

        indices = torch.arange(bs)[~have_seen]
        self.indices = indices

        if selection_method == "random":
            # weights = (self.train_nts == 1).float()
            self.scores = torch.rand(len(indices), device=self.device)
        elif selection_method == "variance":
            # indices = torch.tensor([b for b in range(bs) if self.train_nts[b].item() == 1], device=self.device)
            scores = torch.cat([self._compute_scores(self.pool[indices_split]).sum(dim=-1).cpu() for indices_split in torch.split(indices, self.eval_batch_size)])
            scores[scores < 0] = 0
            self.scores = scores
        elif "stochastic" in selection_method.split("_"):
            # indices = torch.tensor([b for b in range(bs) if self.train_nts[b].item() == 1], device=self.device)
            scores = torch.cat([self._compute_scores(self.pool[indices_split]).sum(dim=-1).cpu() for indices_split in torch.split(indices, self.eval_batch_size)])
            scores[scores < 0] = 0
            scores = torch.log(scores)
            scores = scores * float(selection_method.split("_")[-1])
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores)))
            self.scores = scores + gumbel_noise
        elif "lcmd" in selection_method.split("_"):
            if "hidden" in selection_method.split("_"):
                features = torch.cat([get_features_hidden_trajectory(self.pool[indices_split], self.ensemble, self.L, self.device) for indices_split in torch.split(indices, self.eval_batch_size)])
            elif "ycov" in selection_method.split("_"):
                features = torch.cat([get_features_ycov_trajectory(self.pool[indices_split], self.ensemble, self.L, self.device) for indices_split in torch.split(indices, self.eval_batch_size)])
            else:
                raise ValueError(f"Feature method in {selection_method} not implemented.")
            
            features = features.flatten(start_dim=1)  # [num_candidates, feature_dim]
            
            self.features = features

            # Calculate pairwise distances once
            self.pairwise_distances = torch.cdist(features, features)
            
            # Initialize selected indices and remaining indices
            self.selected_indices = []
            self.remaining_indices = list(range(len(indices)))
        else:
            raise ValueError(f"Initial selection method {selection_method} not implemented.")

    @torch.no_grad()
    def get_next(self):
        selection_method = self.initial_selection_method
        if selection_method in ["random", "variance"] or "stochastic" in selection_method.split("_"):
            top_index = torch.argmax(self.scores)
            self.scores[top_index] = -np.inf
            return self.indices[top_index].item()
        elif "lcmd" in selection_method.split("_"):
            if len(self.remaining_indices) == 0:
                raise ValueError("No more samples to select.")
            if len(self.selected_indices) == 0:
                # Select the point with the largest norm for the first selection
                norms = torch.norm(self.features, dim=1)
                idx = torch.argmax(norms).item()
                self.selected_indices.append(idx)
                self.remaining_indices.remove(idx)
            else:
                # Use pre-computed distances
                distances_to_selected = self.pairwise_distances[self.remaining_indices][:, self.selected_indices]
                
                # Find the closest selected point for each remaining point
                min_distances, closest_selected = torch.min(distances_to_selected, dim=1)
                
                # Group remaining points by their closest selected point
                clusters = [[] for _ in range(len(self.selected_indices))]
                for i, cluster_idx in enumerate(closest_selected):
                    clusters[cluster_idx.item()].append(self.remaining_indices[i])
                
                # Find the largest cluster
                largest_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
                largest_cluster = clusters[largest_cluster_idx]
                
                # Select the point in the largest cluster that's furthest from its center
                cluster_features = self.features[largest_cluster]
                cluster_center = torch.mean(cluster_features, dim=0, keepdim=True)
                cluster_distances = torch.cdist(cluster_features, cluster_center).squeeze()
                furthest_idx = torch.argmax(cluster_distances).item()
                idx = largest_cluster[furthest_idx]
                
                self.selected_indices.append(idx)
                self.remaining_indices.remove(idx)
            
            return self.indices[idx].item()
        else:
            raise ValueError(f"Initial selection method {selection_method} not implemented")

    @torch.no_grad()
    def initial_method(self, index, budget):
        selection_method = self.post_selection_method
        L = self.L
        # print(index)
        X = self.pool[index] # [1, nx]
        X = X.unsqueeze(0)
        bs = X.shape[0] # 1

        if "prior" in selection_method.split("_"):
            scores = self._compute_variance_prior(X) # [1, nt]
        elif "direct" in selection_method.split("_"):
            scores = self._compute_variance_direct(X, mode=self.eer_mode) # [1, nt]
        else:
            raise ValueError(f"Selection method {selection_method} not implemented.")

        scores=scores.cpu() # [1, L]
        scores[scores == float('inf')] = -np.inf
        scores[scores.isnan()] = -np.inf

        costs = torch_expand(torch.arange(L)[None], 0, bs) + 1 # [1, L]
        utility = scores / costs # [1, L]

        # print(utility)
        utility[costs > budget] = -np.inf

        if "max" in selection_method.split("_"):
            max_indices = torch.argmax(utility.flatten())
        elif "stochastic" in selection_method.split("_"):
            utility_temp = utility.clone()
            utility_temp[utility_temp < 0] = 0
            utility_temp = torch.log(utility_temp)
            utility_temp = utility_temp * float(selection_method.split("_")[-1])
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(utility_temp)))
            utility_noisy = utility_temp + gumbel_noise
            max_indices = torch.argmax(utility_noisy.flatten())
        index = max_indices // L
        time = max_indices % L # an index in [0, L)
        time += 1 # an index in [1, L]
        
        S = torch.zeros(1,L).bool()
        S[0,:time] = True
        return S

    @torch.no_grad()
    def flexible_method(self, index, budget):
        selection_method = self.post_selection_method
        L = self.L
        X = self.pool[index] # [1, nx]
        # print(X)
        X = X.unsqueeze(0)
        bs = X.shape[0] # 1

        max_L = min(L, budget)

        if "prior" in selection_method.split("_"):
            scores = self._compute_variance_prior_wo_cumsum(X) # [nt-1]
            # print(scores)
            S = torch.ones(1,L).bool()
            S[:,max_L:] = False
            for t in range(max_L-1, -1, -1): # [L-1, 0]
                if S.sum() == 1:
                    break
                score = (scores * S).sum(dim=1) / S.sum(dim=1)
                S_new = S.clone()
                S_new[0,t] = False
                new_score = (scores * S_new).sum(dim=1) / S_new.sum(dim=1)
                if "max" in selection_method.split("_"):
                    if new_score > score:
                        S = S_new
                elif "stochastic" in selection_method.split("_"):
                    if torch.rand(1) < (new_score / (new_score + score).pow(float(selection_method.split("_")[-1]))).cpu():
                        S = S_new
        elif "approx" in selection_method.split("_"):
            scores = self._compute_variance_prior_wo_cumsum(X).squeeze(0) # [nt-1]
            scores = scores.cpu()
            # append 0 at the start
            scores = torch.cat([torch.zeros(1), scores], dim=0) # [nt]
            scores_diff = scores[1:] - scores[:-1] # [nt-1]
            scores_diff_weighted = scores_diff * torch.arange(L, 0, -1).float() # [nt-1]
            sorted_indices = torch.argsort(scores_diff_weighted, descending=True) # [nt-1]
            sorted_scores = scores_diff_weighted[sorted_indices]
            candidate_scores = sorted_scores.cumsum(dim=0) # [nt-1]
            candidate_scores = (candidate_scores) / torch.arange(1, L+1).float() # [nt-1]
            candidate_scores[candidate_scores < 0] = 0
            if "max" in selection_method.split("_"):
                best_index = torch.argmax(candidate_scores)
            elif "stochastic" in selection_method.split("_"):
                candidate_scores = candidate_scores / candidate_scores.sum()
                candidate_scores = candidate_scores**float(selection_method.split("_")[-1])
                best_index = torch.multinomial(candidate_scores, 1).item()
            else:
                raise ValueError(f"Selection method {selection_method} not implemented.")
            S = torch.zeros(1,L).bool()
            S[:,sorted_indices[:best_index+1]] = True
        elif "direct" in selection_method.split("_"):
            S = torch.ones(1,L).bool()
            S[:,max_L:] = False
            a = EER_Calculator(self.ensemble, X, L, self.eval_batch_size, self.device, mode=self.eer_mode) # mean field eer calculator
            score = a(S) / S.sum()
            for i in range(10):
                S_new = S.clone()
                change = torch.bernoulli(torch.ones(1,L) * 0.1).bool()
                S_new = S_new ^ change
                new_score = a(S_new) / S_new.sum()
                if "max" in selection_method.split("_"):
                    if new_score > score:
                        S = S_new
                        score = new_score
                elif "stochastic" in selection_method.split("_"):
                    # use mcmc
                    if torch.rand(1) < min(1, (new_score / score).cpu()):
                        S = S_new
                        score = new_score
            # for t in range(max_L-1, -1, -1): # [L-1, 0]
            #     if S.sum() == 1:
            #         break
            #     score = a(S) / S.sum()
            #     S_new = S.clone()
            #     S_new[0,t] = False
            #     new_score = a(S_new) / S_new.sum()
            #     if "max" in selection_method.split("_"):
            #         if new_score > score:
            #             S = S_new
            #     elif "stochastic" in selection_method.split("_"):
            #         if torch.rand(1) < (new_score / (new_score + score).pow(float(selection_method.split("_")[-1]))).cpu():
            #             S = S_new
        else:
            raise ValueError(f"Selection method {selection_method} not implemented.")
        
        return S

    @torch.no_grad()
    def post_selection(self, index, budget):
        selection_method = self.post_selection_method
        L = self.L
        if selection_method == 'all':
            S = torch.ones(1,L).bool()
        elif 'initial' in selection_method.split('_'):
            if 'p' in selection_method.split('_'):
                p = float(selection_method.split('_')[-1])
                S_temp = torch.bernoulli(torch.ones(L) * p).bool()
                S = torch.zeros(L).bool()
                S[:S_temp.sum()] = True
                if S.sum() > budget:
                    S = torch.zeros(L).bool()
                    S[:budget] = True
                S = S[None]
            else:
                S = self.initial_method(index, budget)
        elif 'flexible' in selection_method.split('_'):
            if 'p' in selection_method.split('_'):
                p = float(selection_method.split('_')[-1])
                S = torch.bernoulli(torch.ones(L) * p).bool()
                if S.sum() > budget:
                    S = torch.zeros(L).bool()
                    S[torch.multinomial(torch.ones(L), budget)] = True
                S = S[None]
            else:
                S = self.flexible_method(index, budget)
        else:
            raise ValueError(f"Post selection method {selection_method} not implemented.")
        return S
            

    def select(self, budget):
        self.initialize_selection()

        total_cost = 0
        selected = {}
        while total_cost < budget:
            top_index = self.get_next()
            S = self.post_selection(top_index, budget=budget-total_cost) # [1, L]
            if total_cost + S.sum() > budget:
                # pick just the first budget - total_cost True indices of S
                S = S.squeeze()
                S_true = S.nonzero()
                S_temp = torch.zeros_like(S)
                S_temp[S_true[:budget-total_cost]] = True
                S = S_temp.unsqueeze(0)
            total_cost += S.sum()
            assert total_cost <= budget
            assert top_index not in selected
            selected[top_index] = S
        print(selected)
        return selected

    ### Acquisition functions
    #region
    @torch.no_grad()
    def _compute_scores(self, X):
        # starting time: 0, 1, 2, ..., nt-1
        L = self.L

        if len(X) == 0:
            raise ValueError("No samples to compute scores.")
        
        X = X.to(self.device) # [bs, 1, nx]
        assert X.ndim==3

        scores = self._compute_variance(X, timesteps=L) # [bs, nt-1-starting_time]
        return scores

    @torch.no_grad()
    def _compute_variance(self, X, timesteps):
        features = self._compute_features(X, timesteps) # [N, bs, max_timesteps, nx]
        variance = torch.sum(features**2, dim=(0, 3)) # [bs, max_timesteps]
        return variance

    @torch.no_grad()
    def _compute_features(self, X, timesteps):
        if type(timesteps) == int:
            timesteps = torch.ones(X.shape[0], dtype=torch.int64) * timesteps
        if type(timesteps) == list:
            timesteps = torch.tensor(timesteps, dtype=torch.int64)
        ensemble = self.ensemble

        X = X.to(self.device) # [bs, 1, nx]
        timesteps = timesteps.to(self.device) # [bs]

        max_timesteps = timesteps.max()

        pred_trajectory = torch_expand(X.unsqueeze(0), 0, len(ensemble)) # [n_models, bs, 1, nx]
        for t in range(max_timesteps):
            indices = timesteps > t
            if not indices.any():
                break
            
            X_t = pred_trajectory[:, indices, t:t+1] # [n_models, bs_indices, 1, nx]
            pred = []
            with torch.no_grad():
                for i, model in enumerate(ensemble):
                    model.eval()
                    pred.append(split_model(model, self.eval_batch_size)(X_t[i]))
            pred = torch.stack(pred, dim=0) # [n_models, bs_indices, 1, nx]
            pred_with_zeros = torch.zeros(len(ensemble), X.shape[0], 1, *X.shape[2:], device=self.device) # [n_models, bs, 1, nx]
            pred_with_zeros[:, indices] = pred
            pred_trajectory = torch.cat([pred_trajectory, pred_with_zeros], dim=2) # [n_models, bs, t+2, nx]

        # Compute variance across ensemble
        mean = torch.mean(pred_trajectory, dim=0, keepdim=True) # [1, bs, max_timesteps+1, nx]
        features = pred_trajectory - mean # [n_models, bs, max_timesteps+1, nx]
        features /= np.sqrt(len(ensemble)-1) # [n_models, bs, max_timesteps+1, nx]
        features = features.flatten(start_dim=3)  # [n_models, bs, max_timesteps+1, nx]
        return features[:, :, 1:, :] # [n_models, bs, max_timesteps, nx]

    @torch.no_grad()
    def _compute_variance_prior_wo_cumsum(self, X: torch.Tensor):
        # starting_time is a tensor of shape [bs]
        L = self.L
        bs = X.shape[0]
        starting_time = torch.zeros(bs, dtype=torch.int64)
        timesteps = L - starting_time # [bs]
        assert starting_time.shape == (bs,)
        # features_temp = torch.cat([self._compute_features(self.pool[indices_split], timesteps) for indices_split in torch.split(torch.arange(bs), self.eval_batch_size)], dim=1) # [n_models, bs, max_timesteps, nx]
        features_temp = self._compute_features(X, timesteps) # [n_models, bs, max_timesteps, nx]
        features = torch.zeros(len(self.ensemble), bs, L+1, features_temp.shape[-1], device=self.device) # [n_models, bs, nt, nx]
        indices = torch.arange(features.shape[2])[None,:] >= (starting_time+1)[:,None] # [bs, nt]
        indices2 = torch.arange(features_temp.shape[2])[None,:] < timesteps[:,None] # [bs, nt]
        features[:, indices, :] = features_temp[:, indices2, :] # [n_models, bs, nt, nx]
        variance = torch.sum(features**2, dim=(0, 3)) # [bs, nt]
        # scores = torch.cumsum(variance, dim=1) # [bs, nt]
        # return scores[:, 1:] # [bs, nt-1]
        return variance[:, 1:].cpu() # [bs, nt-1]

    @torch.no_grad()
    def _compute_variance_prior(self, X: torch.Tensor):
        # starting_time is a tensor of shape [bs]
        L = self.L
        bs = X.shape[0]
        starting_time = torch.zeros(bs, dtype=torch.int64)
        timesteps = L - starting_time # [bs]
        assert starting_time.shape == (bs,)
        # features_temp = torch.cat([self._compute_features(self.pool[indices_split], timesteps) for indices_split in torch.split(torch.arange(bs), self.eval_batch_size)], dim=1) # [n_models, bs, max_timesteps, nx]
        features_temp = self._compute_features(X, timesteps) # [n_models, bs, max_timesteps, nx]
        features = torch.zeros(len(self.ensemble), bs, L+1, features_temp.shape[-1], device=self.device) # [n_models, bs, nt, nx]
        indices = torch.arange(features.shape[2])[None,:] >= (starting_time+1)[:,None] # [bs, nt]
        indices2 = torch.arange(features_temp.shape[2])[None,:] < timesteps[:,None] # [bs, nt]
        features[:, indices, :] = features_temp[:, indices2, :] # [n_models, bs, nt, nx]
        variance = torch.sum(features**2, dim=(0, 3)) # [bs, nt]
        scores = torch.cumsum(variance, dim=1) # [bs, nt]
        return scores[:, 1:] # [bs, nt-1]

    @torch.no_grad()
    def _compute_variance_direct(self, X: torch.Tensor, mode='EER'):
        L = self.L
        bs = X.shape[0]
        ensemble = self.ensemble

        # X = self.Y[:, 0].unsqueeze(1)
        a = EER_Calculator(ensemble, X, L, self.eval_batch_size, self.device, mode=mode) # mean field eer calculator
        S = torch.zeros(bs, L, device=self.device).bool()
        scores = []
        for i in range(L):
            S_i = S.clone()
            S_i[:, :i+1] = True
            score = a(S_i)
            scores.append(score)
        scores = torch.stack(scores, dim=1) # [bs, L]

        return scores
    
    #endregion