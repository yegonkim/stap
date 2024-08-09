#!/bin/bash

python run_experiment_al.py --equation $EQ --experiment trajectory_variance --unrolling $UNROLL
python run_experiment_al.py --equation $EQ --experiment trajectory_lcmd --unrolling $UNROLL
