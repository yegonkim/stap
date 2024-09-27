# ablation: longer trajectories

for SEED in 0 1 2
do
    for EQ in NS_long
    do
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_9_27 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_9_27 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all
        python run_experiment_al_flexible.py task=$EQ seed=$SEED wandb.project=${EQ}_9_27 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max
    done
done

