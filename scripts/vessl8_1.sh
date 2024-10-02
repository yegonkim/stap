# ablation: longer trajectories

for SEED in 0 1 2
do
    for EQ in KS_short_27
    do
        # python run_experiment_al_flexible.py wandb.project=KS_32_16 task=$EQ seed=$SEED initial_datasize=32 batch_acquire=16 num_acquire=5 initial_selection_method=random post_selection_method=all
        python run_experiment_al_flexible.py wandb.project=KS_32_16 task=$EQ seed=$SEED initial_datasize=32 batch_acquire=16 num_acquire=5 initial_selection_method=random post_selection_method=flexible_max
        # python run_experiment_al_flexible.py wandb.project=KS_32_16 task=$EQ seed=$SEED initial_datasize=32 batch_acquire=16 num_acquire=5 initial_selection_method=variance post_selection_method=all
        # python run_experiment_al_flexible.py wandb.project=KS_32_16 task=$EQ seed=$SEED initial_datasize=32 batch_acquire=16 num_acquire=5 initial_selection_method=variance post_selection_method=flexible_max
    done
done

