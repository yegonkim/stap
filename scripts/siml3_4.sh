for SEED in 1000
do
    for EQ in KdV
    do
        python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_24_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max num_proposal=10 filter_method=ignore
        # python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_24_flexible_32 initial_datasize=32 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max num_proposal=100 filter_method=ignore
    done
done

# for SEED in 1000
# do
#     for EQ in KdV
#     do
#         python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_24_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max num_proposal=10 filter_method=ignore
#         python run_experiment_al_flexible.py task=$EQ seed=$SEED nt=14 wandb.project=${EQ}_9_24_flexible_128 initial_datasize=128 batch_acquire=8 num_acquire=10 initial_selection_method=stochastic_1.0 post_selection_method=flexible_max num_proposal=100 filter_method=ignore
#     done
# done