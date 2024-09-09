#!/bin/bash
EQ='KdV'

for SEED in 0 1 2 3 4
do
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=0 num_acquire=10 timesteps=14 initial_datasize=30 batch_acquire=30 
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=0 num_acquire=10 timesteps=7 initial_datasize=65 batch_acquire=65
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=0 num_acquire=10 timesteps=2 initial_datasize=390 batch_acquire=390

    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=60 num_acquire=10 timesteps=14 initial_datasize=0 batch_acquire=30 
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=60 num_acquire=10 timesteps=7 initial_datasize=0 batch_acquire=65
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=60 num_acquire=10 timesteps=2 initial_datasize=0 batch_acquire=390

    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=120 num_acquire=10 timesteps=14 initial_datasize=0 batch_acquire=30 
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=120 num_acquire=10 timesteps=7 initial_datasize=0 batch_acquire=65
    python run_experiment_whole_part.py task=$EQ seed=$SEED wandb.project=${EQ}_stable_mse_9_8 datasize=10000 whole_initial_datasize=120 num_acquire=10 timesteps=2 initial_datasize=0 batch_acquire=390
done
