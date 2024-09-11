#!/bin/bash

for SEED in 10 11 12 13 14
do
    for EQ in KdV Burgers KS
    do
        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_11 nt=131 initial_datasize=32 ensemble_size=5 scenario=fixed initial_selection_method=random
        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_11 nt=131 initial_datasize=32 ensemble_size=5 scenario=fixed initial_selection_method=stochastic
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_11 nt=131 initial_datasize=32 ensemble_size=5 scenario=flexible flexible_selection_method=single_zero_variance_direct_stochastic
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_11 nt=131 initial_datasize=32 ensemble_size=5 scenario=flexible flexible_selection_method=single_zero_mutual_exp_stochastic
    done
done
