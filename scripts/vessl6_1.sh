# ablation: longer trajectories

for SEED in 0 1 2 3 4
do
    for EQ in KS_short_27
    do
        python run_experiment_al_flexible.py wandb.project=${EQ}_rmse task=$EQ seed=$SEED initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max loss_function=rmse
    done
done

