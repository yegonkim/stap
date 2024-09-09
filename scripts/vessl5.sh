#!/bin/bash

for SEED in 0 1 2 3 4
do
    for EQ in KdV Burgers KS
    do
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=fixed initial_selection_method=random ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=fixed initial_selection_method=variance ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=fixed initial_selection_method=stochastic ensemble_size=2

        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_zero_variance_prior_max ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_zero_variance_direct_max ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_zero_variance_prior_stochastic ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_zero_variance_direct_stochastic ensemble_size=2

        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_last_variance_prior_max ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_last_variance_direct_max ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_last_variance_prior_stochastic ensemble_size=2
        # python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_last_variance_direct_stochastic ensemble_size=2

        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_ignore_variance_prior_max ensemble_size=2
        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_ignore_variance_direct_max ensemble_size=2
        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_ignore_variance_prior_stochastic ensemble_size=2
        python run_experiment_al_time_batch.py task=$EQ seed=$SEED wandb.project=${EQ}_time_batch_9_10 scenario=flexible flexible_selection_method=single_ignore_variance_direct_stochastic ensemble_size=2
    done
done
