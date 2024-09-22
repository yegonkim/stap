# #!/bin/bash

for SEED in 1000 1001
do
    for EQ in NS KdV Heat
    do
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_max
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_stochastic_1.0
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_p_0.5 cheat=True
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_max filter=True
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_stochastic_1.0 filter=True
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_approx_stochastic_1.0 filter=1.5
    done
done

for SEED in 1000 1001
do
    for EQ in NS KdV Heat
    do
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_max
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_stochastic_1.0
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_p_0.5 cheat=True
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_max filter=True
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_direct_stochastic_1.0 filter=True
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_approx_stochastic_1.0 filter=1.5
    done
done