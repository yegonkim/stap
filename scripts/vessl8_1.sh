# #!/bin/bash

# for SEED in 100 101 102 103 104
# do
#     for EQ in Heat
#     do
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_prior_max
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_prior_stochastic_1.0
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_approx_max
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_approx_stochastic_1.0
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_max
#         python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_best_9_20 nt=14 exponential_data=False initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_stochastic_1.0
#     done
# done
