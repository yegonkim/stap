#!/bin/bash
for SEED in 0 1 2 3 4
do
    for EQ in 'KdV' 'Burgers' 'KS'
    do
        for SIZE in 32 64 128 256 512
        do
            python run_experiment_al.py task=$EQ selection_method=random ensemble_size=2 seed=$SEED initial_datasize=$SIZE batch_acquire=32
            # python run_experiment_al.py task=$EQ selection_method=variance ensemble_size=2 seed=$SEED initial_datasize=$SIZE batch_acquire=32
            # python run_experiment_al.py task=$EQ selection_method=stochastic ensemble_size=2 seed=$SEED initial_datasize=$SIZE batch_acquire=32
        done
    done
done

