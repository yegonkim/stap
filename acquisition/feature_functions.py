# torch
import torch
import numpy as np
from utils import normalized_model, normalized_residual_model

@torch.no_grad()
def get_features_ycov(X, ensemble):
    with torch.no_grad():
        features = []
        for model in ensemble:
            with torch.no_grad():
                model.eval()
                pred = torch.cat([model(x) for x in X.split(256, dim=0)], dim=0) # [bs, ...]
                features.append(pred)
        features = torch.stack(features, dim=1) # [bs, N, ...]
        features = features - torch.mean(features, dim=1, keepdim=True) # [bs, N, ...]; centering
        features = features / np.sqrt(len(ensemble)-1) # [bs, N, ...]; normalization by 1/sqrt(N-1)
        features = torch.flatten(features, start_dim=2) # [bs, N, dim]; vectorize
        return features # [bs, N, dim]

@torch.no_grad()
def get_features_ycov_trajectory(X, ensemble, num_steps, device):
    X = X.to(device)
    all_features = []
    sketch_dim = 512

    for i in range(num_steps):
        features = []
        for model in ensemble:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(x) for x in X.split(256, dim=0)], dim=0) # [bs, ...]
                features.append(pred)
        features = torch.stack(features, dim=1) # [bs, N, ...]
        features = features - torch.mean(features, dim=1, keepdim=True) # [bs, N, ...]; centering
        features = features / np.sqrt(len(ensemble)-1) # [bs, N, ...]; normalization by 1/sqrt(N-1)
        features = torch.flatten(features, start_dim=2) # [bs, N, dim]; vectorize
        all_features.append(features)
    all_features = torch.stack(all_features, dim=-1) # [bs, N, dim, num_steps]
    all_features = all_features.flatten(start_dim=1) # [bs, N*dim*num_steps]
    U = torch.randn(sketch_dim, all_features.shape[1], device=all_features.device) # [sketch_dim, N*dim*num_steps]
    all_features = torch.einsum('ij,bj->bi', U, all_features) # [bs, sketch_dim]
    return all_features # [bs, sketch_dim]


@torch.no_grad()
def get_features_hidden(X, ensemble):
    assert len(ensemble) == 1
    model = ensemble[0]
    model.eval()

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    model.projection.fcs[0].register_forward_hook(get_activation('hidden'))

    with torch.no_grad():
        features = []
        for x in X.split(256, dim=0):
            output = model(x)
            features.append(activations['hidden'])
        features = torch.cat(features, dim=0) # [bs, channel_size, dim]
        features = features.flatten(start_dim=2)
    
    return features # [bs, channel_size, dim]

@torch.no_grad()
def get_features_hidden_trajectory(X, ensemble, num_steps, device):
    X = X.to(device)
    features_list = []
    for model in ensemble:
        model.eval()
        sketch_dim = 512

        ######
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        if isinstance(model, normalized_model) or isinstance(model, normalized_residual_model):
            model.model.projection.fcs[0].register_forward_hook(get_activation('hidden'))
        else:
            model.projection.fcs[0].register_forward_hook(get_activation('hidden'))
        ######


        all_features = []
        for i in range(num_steps):
            U=None
            features = []
            outputs = []
            model.eval()
            for x in X.split(256, dim=0):
                output = model(x)
                feature = activations['hidden'].flatten(start_dim=1) # [bs, dim]
                U = torch.randn(sketch_dim, feature.shape[1], device=feature.device) if U is None else U # [sketch_dim, dim]
                sketched_feature = torch.einsum('ij,bj->bi', U, feature) # [bs, sketch_dim]
                outputs.append(output)
                features.append(sketched_feature)
            outputs = torch.cat(outputs, dim=0) # [bs, ...]
            X = outputs
            features = torch.cat(features, dim=0) # [bs, sketch_dim]
            all_features.append(features)
        all_features = torch.stack(all_features, dim=-1).sum(dim=-1) # [bs, sketch_dim]
        assert all_features.ndim == 2 and all_features.shape[1] == sketch_dim
        all_features /= np.sqrt(sketch_dim)
        features_list.append(all_features)
    all_features = torch.cat(features_list, dim=1) # [bs, N*sketch_dim]
    U = torch.randn(sketch_dim, all_features.shape[1], device=all_features.device)
    all_features = torch.einsum('ij,bj->bi', U, all_features) # [bs, sketch_dim]

    return all_features # [bs, channel_size, dim*num_steps]
