#!/bin/bash

for SEED in 10 11 12 13 14
do
    for EQ in KdV
    do
        # python run_experiment_long_front.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_12 nt=131 initial_datasize=16 p=1.0 whole_initial_datasize=64 dataset.train_path=data/KdV_train_10000_default.h5 dataset.test_path=data/KdV_test_10000_default.h5 wandb.use=False
        python run_experiment_long_front.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_12 nt=131 initial_datasize=32 p=0.5 whole_initial_datasize=64 dataset.train_path=data/KdV_train_10000_default.h5 dataset.test_path=data/KdV_test_10000_default.h5 wandb.use=False
        # python run_experiment_long_front.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_12 nt=131 initial_datasize=64 p=0.25 whole_initial_datasize=64 dataset.train_path=data/KdV_train_10000_default.h5 dataset.test_path=data/KdV_test_10000_default.h5 wandb.use=False
        # python run_experiment_long_front.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_12 nt=131 initial_datasize=128 p=0.125 whole_initial_datasize=64 dataset.train_path=data/KdV_train_10000_default.h5 dataset.test_path=data/KdV_test_10000_default.h5 wandb.use=False
        # python run_experiment_long_front.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_12 nt=131 initial_datasize=256 p=0.0625 whole_initial_datasize=64 dataset.train_path=data/KdV_train_10000_default.h5 dataset.test_path=data/KdV_test_10000_default.h5 wandb.use=False
        # python run_experiment_long_front.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_12 nt=131 initial_datasize=512 p=0.03125 whole_initial_datasize=64 dataset.train_path=data/KdV_train_10000_default.h5 dataset.test_path=data/KdV_test_10000_default.h5 wandb.use=False
    done
done
