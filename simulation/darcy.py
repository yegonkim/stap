import numpy as np
import string, random, os, time
import scipy
import torch_dct as dct
import subprocess
import torch

# from hdf5storage import savemat
from hdf5storage import loadmat, savemat

import sys

import wandb

from .sim import Sim


def get_random_alphanumeric_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str

# % Return a sample of a Gaussian random field on [0,1]^2 with: 
# %       mean 0
# %       covariance operator C = (-Delta + tau^2)^(-alpha)
# % where Delta is the Laplacian with zero Neumann boundary conditions.
def GRF(xi, alpha=2, tau=3):
    # Random variables in KL expansion
    # xi = np.random.normal(0, 1, (s, s))
    s = xi.shape[0]
    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='xy')
    K1, K2 = K1.to(xi.device), K2.to(xi.device)
    coef = tau**(alpha-1) * (torch.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha/2)
    # Set the first coefficient to 0 to account for mean 0
    coef[0, 0] = 0
    
    # Construct the KL coefficients
    L = s * coef * xi
    
    # Perform the inverse discrete cosine transform
    # U = idct(idct(L, axis=0, norm='ortho'), axis=1, norm='ortho')
    U = dct.idct_2d(L, norm='ortho')
    
    return U


class Darcy(Sim):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.ndim = 2
        self.fid = 32
        self.ubound = 2.0
        self.lbound = -2.0
        self.cfg = cfg
        self.param_dim = self.fid * self.fid

    def query_in_unnorm(self, params):
        # params = xi : bs x s x s where s is the highest fidelity
        bs = params.shape[:-1]
        params = params.view(-1, self.fid, self.fid)
        # X = []
        #print(bs, s, fid)
        # for i in range(params.shape[0]):
        #     xi = params[i]
        #     u = GRF(xi)
        #     X.append(u)
        X = torch.vmap(GRF)(params) # [bs, fid, fid]
        # X = torch.stack(X, dim=0) # [bs, s, s]
        # X = torch.nn.functional.interpolate(X[:,None,:,:], size=fid, mode='area').squeeze(1) # [bs, fid, fid]

        X = torch.exp(X)
        # X = torch.sign(X) * 4 + 8
        #print(X.shape)
        #X = X.view(bs, *s)
        X = X.view(*bs, self.fid, self.fid)
        return X
        
    def query_out_unnorm(self, X):
        # params = [xi] : bs x s x s
        bs = X.shape[:-2]
        X = X.view(-1, self.fid, self.fid)

        if wandb.run is not None:
            datapath = os.path.join(wandb.run.dir, 'data', '__buff__')
        else:
            datapath = 'data/__buff__'
        if not os.path.exists(datapath):
            os.makedirs(datapath)

        X = X.detach().cpu().numpy()

        query_key = get_random_alphanumeric_string(77)
        input_path = os.path.join(datapath, query_key + '.mat')
        query_key = get_random_alphanumeric_string(77)
        buff_path = os.path.join(datapath, query_key + '.mat')

        # savemat(input_path, {'params': params, 'fid': float(self.fidelity_list[m])})

        savemat(input_path, {'X': X})
        
        matlab_cmd = 'addpath(genpath(\'simulation/Darcy\'));'
        matlab_cmd += f'query_client_darcy(\'{input_path}\', \'{buff_path}\');'
        matlab_cmd += 'quit force'

        print('querying...')
        command = ["matlab", "-nodesktop", "-r", matlab_cmd]
        # process = subprocess.Popen(command,
        #                      stdout=subprocess.PIPE, 
        #                      stderr=subprocess.PIPE,
        #                      )
        process = subprocess.Popen(command,
                             stdout=sys.stdout,
                             stderr=sys.stdout,
        )
        process.wait()
        print('done!')
        
        retrived_data = loadmat(buff_path, squeeze_me=True, struct_as_record=False, mat_dtype=True)['data']
        # delete the temporary files
        os.remove(input_path)
        os.remove(buff_path)
        Y = retrived_data.Y # [bs, s, s]
        Y = torch.tensor(Y, dtype=torch.float32)
        if Y.ndim<3: # this happens if bs=1
            Y = Y.unsqueeze(0)
        
        Y = Y.view(*bs, self.fid, self.fid)
        return Y