

Generate data

python generate_data.py --experiment=KdV --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=Burgers --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --nt=180 --end_time=18. --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=KS --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=64 --nt=500 --suffix default --batch_size 4096 --device cuda

python generate_data.py --experiment=KdV --train_samples=10000 --valid_samples=0 --test_samples=10000 --L=128 --end_time=56. --suffix default --batch_size 10000 --device cuda --nt 140 --nt_effective 140
python generate_data.py --experiment=Burgers --train_samples=10000 --valid_samples=0 --test_samples=10000 --L=128 --end_time=14. --suffix default --batch_size 4096 --device cuda --nt 140 --nt_effective 140
python generate_data.py --experiment=KS --train_samples=1000 --valid_samples=0 --test_samples=1000 --L=64 --end_time=28. --suffix default --batch_size 4096 --device cuda --nt 140 --nt_effective 140


Run experiment

python run_experiment.py --equation KdV --experiment direct
python run_experiment.py --equation KdV --experiment multi
python run_experiment.py --equation KdV --experiment ar_0
python run_experiment.py --equation KdV --experiment ar_1

python run_experiment.py --equation Burgers --experiment direct
python run_experiment.py --equation Burgers --experiment multi
python run_experiment.py --equation Burgers --experiment ar_0
python run_experiment.py --equation Burgers --experiment ar_1

python run_experiment.py --equation KS --experiment direct
python run_experiment.py --equation KS --experiment multi
python run_experiment.py --equation KS --experiment ar_0
python run_experiment.py --equation KS --experiment ar_1

python run_experiment.py --equation KS --experiment ar_1

AL experiments

python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0
python run_experiment_al.py --equation KdV --experiment direct_variance --unrolling 0
python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0
python run_experiment_al.py --equation KdV --experiment trajectory_variance --unrolling 0
python run_experiment_al.py --equation KdV --experiment trajectory_lcmd --unrolling 0

python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1
python run_experiment_al.py --equation Burgers --experiment direct_variance --unrolling 1
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1
python run_experiment_al.py --equation Burgers --experiment trajectory_variance --unrolling 1
python run_experiment_al.py --equation Burgers --experiment trajectory_lcmd --unrolling 1

python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0
python run_experiment_al.py --equation KS --experiment direct_variance --unrolling 0
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0
python run_experiment_al.py --equation KS --experiment trajectory_variance --unrolling 0
python run_experiment_al.py --equation KS --experiment trajectory_lcmd --unrolling 0

AL_comb

python run_experiment_al_comb.py --equation KdV --experiment direct_variance --unrolling 0
python run_experiment_al_comb.py --equation KdV --experiment direct_lcmd --unrolling 0
python run_experiment_al_comb.py --equation Burgers --experiment direct_variance --unrolling 1
python run_experiment_al_comb.py --equation Burgers --experiment direct_lcmd --unrolling 1
python run_experiment_al_comb.py --equation KS --experiment direct_variance --unrolling 0
python run_experiment_al_comb.py --equation KS --experiment direct_lcmd --unrolling 0

python run_experiment_al_comb.py --equation KdV --experiment trajectory_variance --unrolling 0
python run_experiment_al_comb.py --equation KdV --experiment trajectory_lcmd --unrolling 0
python run_experiment_al_comb.py --equation Burgers --experiment trajectory_variance --unrolling 1
python run_experiment_al_comb.py --equation Burgers --experiment trajectory_lcmd --unrolling 1
python run_experiment_al_comb.py --equation KS --experiment trajectory_variance --unrolling 0
python run_experiment_al_comb.py --equation KS --experiment trajectory_lcmd --unrolling 0

sca python run_experiment_al_comb.py --equation KdV --experiment trajectory_variance --unrolling 0
sca python run_experiment_al_comb.py --equation KdV --experiment trajectory_lcmd --unrolling 0
sca python run_experiment_al_comb.py --equation Burgers --experiment trajectory_variance --unrolling 1
sca python run_experiment_al_comb.py --equation Burgers --experiment trajectory_lcmd --unrolling 1
sca python run_experiment_al_comb.py --equation KS --experiment trajectory_variance --unrolling 0
sca python run_experiment_al_comb.py --equation KS --experiment trajectory_lcmd --unrolling 0

export EQ=KdV
export UNROLL=0

export EQ=Burgers
export UNROLL=1

export EQ=KS
export UNROLL=0

bash tmux_scripts/al.sh
bash tmux_scripts/al_2.sh
bash tmux_scripts/al_comb.sh
bash tmux_scripts/al_comb_2.sh


python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0 --batch_acquire 8
python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0 --batch_acquire 16
python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0 --batch_acquire 32
python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0 --batch_acquire 64
python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0 --batch_acquire 128
python run_experiment_al.py --equation KdV --experiment direct_lcmd --unrolling 0 --batch_acquire 256

python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1 --batch_acquire 8
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1 --batch_acquire 16
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1 --batch_acquire 32
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1 --batch_acquire 64
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1 --batch_acquire 128
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1 --batch_acquire 256

python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0 --batch_acquire 8
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0 --batch_acquire 16
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0 --batch_acquire 32
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0 --batch_acquire 64
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0 --batch_acquire 128
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0 --batch_acquire 256



python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0 --batch_acquire 8
python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0 --batch_acquire 16
python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0 --batch_acquire 32
python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0 --batch_acquire 64
python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0 --batch_acquire 128
python run_experiment_al.py --equation KdV --experiment direct_random --unrolling 0 --batch_acquire 256

python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1 --batch_acquire 8
python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1 --batch_acquire 16
python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1 --batch_acquire 32
python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1 --batch_acquire 64
python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1 --batch_acquire 128
python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1 --batch_acquire 256

python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0 --batch_acquire 8
python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0 --batch_acquire 16
python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0 --batch_acquire 32
python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0 --batch_acquire 64
python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0 --batch_acquire 128
python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0 --batch_acquire 256

