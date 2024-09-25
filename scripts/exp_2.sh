# ablation: different base methods

for SEED in 0 1 2 3
do
    for EQ in Heat KdV NS
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=flexible_max comment=base
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=variance post_selection_method=flexible_max comment=base
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=lcmd_hidden post_selection_method=flexible_max comment=base
    done
done

