#!/bin/bash

python run_experiment_al_comb.py --equation $EQ --experiment direct_random --unrolling $UNROLL
python run_experiment_al_comb.py --equation $EQ --experiment direct_variance --unrolling $UNROLL
python run_experiment_al_comb.py --equation $EQ --experiment direct_lcmd --unrolling $UNROLL
