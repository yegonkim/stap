LUPI with neural operators


VESSL init
conda init bash
conda activate /mnt/home/yegonkim/miniconda3/envs/lupi
cp -r /mnt/home/yegonkim/home/lupi/data data

Generate data

python generate_data.py --experiment=KdV --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --suffix default --batch_size 4096 --device cuda

python generate_data.py --experiment=KS --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=64 --nt=500 --suffix default --batch_size 4096 --device cuda

python generate_data.py --experiment=Burgers --train_samples=1024 --valid_samples=1024 --test_samples=4096 --end_time=18. --nt=180 --suffix default --batch_size 4096 --device cuda


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
python run_experiment_al.py --equation KdV --experiment trajectory_random --unrolling 0
python run_experiment_al.py --equation KdV --experiment trajectory_variance --unrolling 0
python run_experiment_al.py --equation KdV --experiment trajectory_lcmd --unrolling 0

python run_experiment_al.py --equation Burgers --experiment direct_random --unrolling 1
python run_experiment_al.py --equation Burgers --experiment direct_variance --unrolling 1
python run_experiment_al.py --equation Burgers --experiment direct_lcmd --unrolling 1
python run_experiment_al.py --equation Burgers --experiment trajectory_random --unrolling 1
python run_experiment_al.py --equation Burgers --experiment trajectory_variance --unrolling 1
python run_experiment_al.py --equation Burgers --experiment trajectory_lcmd --unrolling 1

python run_experiment_al.py --equation KS --experiment direct_random --unrolling 0
python run_experiment_al.py --equation KS --experiment direct_variance --unrolling 0
python run_experiment_al.py --equation KS --experiment direct_lcmd --unrolling 0
python run_experiment_al.py --equation KS --experiment trajectory_random --unrolling 0
python run_experiment_al.py --equation KS --experiment trajectory_variance --unrolling 0
python run_experiment_al.py --equation KS --experiment trajectory_lcmd --unrolling 0