#!/bin/bash

for SEED in 10
do
    for EQ in Heat KdV KS
    do
        # python run_experiment_long.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_14 nt=131 initial_datasize=32 num_acquire=4 p=0.5
        # python run_experiment_long.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_14 nt=131 initial_datasize=16 num_acquire=4 p=1.0
        python run_experiment_long.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_14 nt=131 initial_datasize=64 num_acquire=4 p=0.25
        # python run_experiment_long.py task=$EQ seed=$SEED wandb.project=${EQ}_long_9_14 nt=131 initial_datasize=128 num_acquire=4 p=0.125
    done
done
