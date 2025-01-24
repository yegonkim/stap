# ablation: longer trajectories

for SEED in 0 1 2 3 4
do
    for EQ in NS_rey_27
    do
        # python run_experiment_al_flexible.py wandb.project=${EQ}_200_epochs task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all train.epochs=200
        python run_experiment_al_flexible.py wandb.project=${EQ}_200_epochs task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=flexible_p_0.5 train.epochs=200
    done
done
