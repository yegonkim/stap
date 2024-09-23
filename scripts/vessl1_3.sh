for SEED in 1000
do
    for EQ in Burgers
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_23_flexible_32_test initial_datasize=48 batch_acquire=8 num_acquire=10 initial_selection_method=random post_selection_method=all cheat=True
    done
done