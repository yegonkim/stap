# ablation: longer trajectories

for SEED in 0 1 2 3 4
do
    for EQ in CNS2
    do
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=lcmd_hidden post_selection_method=all
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=variance post_selection_method=all
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_p_0.5
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_p_0.25
        python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_p_0.125
        # python run_experiment_al_flexible.py wandb.project=${EQ}_final task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_p_0.0625
    done
done
