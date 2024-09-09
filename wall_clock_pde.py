import torch
import time
from matplotlib import pyplot as plt
import math

from generate_data import main

class Args:
    def __init__(self):
        self.experiment = 'KdV'
        self.device = 'cpu'
        self.end_time = 100.
        self.nt = 250
        self.nt_effective = 140
        self.nx = 256
        self.L = 128.
        self.train_samples = 0
        self.valid_samples = 0
        self.test_samples = 0
        self.batch_size = 1
        self.suffix = ''
        self.log = False
        self.solver = 'dopri5'
        self.nu = 0.01

args = Args()
args.experiment = 'KdV'
args.train_samples = 16
args.valid_samples = 0
args.test_samples = 0
args.L = 128.
args.suffix = 'default'
args.batch_size = 1000000
args.device = 'cuda'

# for train_samples in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
#     print(f"train_samples: {train_samples}")
#     args.train_samples = train_samples
#     main(args)

train_samples_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
train_samples_list = [math.floor(1.5 * i) for i in train_samples_list]

for train_samples in train_samples_list:
    print(f"train_samples: {train_samples}")
    args.train_samples = train_samples
    main(args)