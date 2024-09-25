# ablation: batch acquire

for SEED in 0 1 2 3
do
    for EQ in Heat KdV NS
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=16 num_acquire=10 initial_selection_method=random post_selection_method=all comment=batch
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=16 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all comment=batch
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=16 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max comment=batch
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=32 num_acquire=10 initial_selection_method=random post_selection_method=all comment=batch
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=32 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all comment=batch
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=32 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max comment=batch
    done
done
