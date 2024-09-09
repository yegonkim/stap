# torch
import torch
import numpy as np

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


def get_features_hidden(X, ensemble):
    assert len(ensemble) == 1
    model = ensemble[0]
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    model.projection.fcs[0].register_forward_hook(get_activation('hidden'))

    with torch.no_grad():
        features = []
        model.eval()
        for x in X.split(256, dim=0):
            output = model(x)
            features.append(activations['hidden'])
        features = torch.cat(features, dim=0) # [bs, channel_size, dim]
        features = features.flatten(start_dim=2)
    
    return features # [bs, channel_size, dim]

# def get_features_hidden(X, ensemble):
#     assert len(ensemble) == 1
#     direct_model = ensemble[0]
#     model = direct_model.model
#     activations = {}

#     def get_activation(name):
#         def hook(model, input, output):
#             activations[name] = output.detach()
#         return hook

#     model.projection.fcs[0].register_forward_hook(get_activation('hidden'))

#     with torch.no_grad():
#         features = []
#         direct_model.eval()
#         for x in X.split(32, dim=0):
#             output = direct_model(x)
#             features.append(activations['hidden'])
#         features = torch.cat(features, dim=0) # [bs, channel_size, dim]
#         features = features.flatten(start_dim=2)
    
#     return features # [bs, channel_size, dim]