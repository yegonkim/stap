# #!/bin/bash

# for SEED in 1000 1001 1002 1003 1004
# do
#     for EQ in KdV Heat
#     do
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_long_9_21_flexible p=1.0 initial_datasize=0 num_acquire=2 post_selection_method=flexible_p_1.0 cheat=True
#         # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_long_9_21_flexible p=0.5 initial_datasize=64 num_acquire=2 post_selection_method=flexible_p_0.5 cheat=True
#         python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_long_9_21_flexible initial_datasize=128 num_acquire=2 post_selection_method=flexible_p_0.25 cheat=True
#     done
# done
