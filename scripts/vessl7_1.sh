#!/bin/bash

for SEED in 10 11 12 13 14
do
    for EQ in KdV KS Heat
    do
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=fixed initial_selection_method=random ensemble_size=1
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=fixed initial_selection_method=variance
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=fixed initial_selection_method=stochastic
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=fixed initial_selection_method=lcmd_ycov
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=fixed initial_selection_method=lcmd_hidden ensemble_size=1
        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=flexible flexible_selection_method=single_zero_variance_direct_stochastic
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_best131_9_14 nt=131 initial_datasize=32 batch_acquire=8 scenario=flexible flexible_selection_method=single_zero_mutual_exp_stochastic
    done
done
