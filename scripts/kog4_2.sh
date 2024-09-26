# ablation: initial datasize

for SEED in 0 1 2 3
do
    for EQ in KdV
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max filter=10 comment=filtering_old
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=lcmd_hidden post_selection_method=all comment=initial
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=lcmd_hidden post_selection_method=flexible_max comment=initial
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=variance post_selection_method=all comment=initial
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=variance post_selection_method=flexible_max comment=initial
    done
done
