#!/bin/bash
EQ='KdV'

for SEED in 0 1 2 3 4
do
    # python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=random
    # python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=variance
    # python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=stochastic
    # python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=lcmd_tp_ycov_individual_max
    # python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=lcmd_tp_hidden_concat ensemble_size=1
    # python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=lcmd_tp_ycov_individual_max
    python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=mutual_individual_max
    python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=mutual_individual_stochastic
    python run_experiment_al.py task=$EQ seed=$SEED wandb.project=${EQ}_onestepal_9_8 selection_method=mutual_concat
done
