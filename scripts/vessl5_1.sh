# for SEED in 1000
# do
#     for EQ in Heat
#     do
#         python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_24_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all train.unrolling=1 comment=fixed_iter
#     done
# done

for SEED in 1000
do
    for EQ in Heat
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_24_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=all train.unrolling=1 comment=fixed_iter
    done
done