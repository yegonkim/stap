#!/bin/bash
EQ='KS'

for SEED in 0 1 2 3 4
do
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=0 num_acquire=2 timesteps=131 initial_datasize=1 batch_acquire=1 
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=0 num_acquire=2 timesteps=66 initial_datasize=2 batch_acquire=2
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=0 num_acquire=2 timesteps=2 initial_datasize=130 batch_acquire=130

    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=60 num_acquire=2 timesteps=131 initial_datasize=0 batch_acquire=1 
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=60 num_acquire=2 timesteps=66 initial_datasize=0 batch_acquire=2
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=60 num_acquire=2 timesteps=2 initial_datasize=0 batch_acquire=130

    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=120 num_acquire=2 timesteps=131 initial_datasize=0 batch_acquire=1
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=120 num_acquire=2 timesteps=66 initial_datasize=0 batch_acquire=2
    python run_experiment_whole_part.py task=$EQ seed=$SEED nt=131 wandb.project=${EQ}_whole_part_9_10 datasize=1000 whole_initial_datasize=120 num_acquire=2 timesteps=2 initial_datasize=0 batch_acquire=130
done
