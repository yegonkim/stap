## Generate data

```
python generate_data.py task=Burgers
python generate_data.py task=KdV
python generate_data.py task=KS
python generate_data.py task=INS
python generate_data.py task=CNS
```

## Run experiment

BASE_METHOD is the base AL method:
* random (Random)
* variance (QbC)
* stochastic_1.0 (SBAL)
* lcmd_hidden (LCMD)

TIMESTEP_SELECTION_METHOD is the timestep selection method:
* all (Full trajectory)
* flexible_max (+STAP)

```
for SEED in 0 1 2 3 4; do
    for EQ in Burgers KdV KS INS CNS; do
        for BASE_METHOD in random variance stochastic_1.0 lcmd_hidden; do
            for TIMESTEP_SELECTION_METHOD in all flexible_max; do
                python run_experiment_al_flexible.py task=$EQ seed=$SEED \
                initial_datasize=32 batch_acquire=8 num_acquire=10 \
                initial_selection_method=$BASE_METHOD \
                post_selection_method=$TIMESTEP_SELECTION_METHOD
            done
        done
    done
done
```

