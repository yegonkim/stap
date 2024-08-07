# torch
import torch
import numpy as np

def get_features_ycov(X, ensemble):
    # P: [bs, dim]
    features = []
    for model in ensemble:
        model.eval()
        pred = torch.cat([model(x) for x in X.split(32, dim=0)], dim=0) # [bs, ...]
        features.append(pred)
    features = torch.stack(features, dim=1) # [bs, N, ...]
    features = features - torch.mean(features, dim=1, keepdim=True) # [bs, N, ...]; centering
    features = features / np.sqrt(len(ensemble)) # [bs, N, ...]; normalization by 1/sqrt(N)
    features = torch.flatten(features, start_dim=2) # [bs, N, dim]; vectorize
    return features # [bs, N, dim]
