# ablation: pushforward, gaussian

for SEED in 0 1 2 3
do
    for EQ in Heat KdV NS
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all train.unrolling=1 comment=training
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all train.unrolling=2 comment=training
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_25 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all train.gaussian_noise=0.01 comment=training
    done
done

