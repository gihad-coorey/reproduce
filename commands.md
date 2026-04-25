- rsync -avz ../reproduce/ gcoorey@kaya01.hpc.uwa.edu.au:/group/pmc086/gcoorey/reproduce

- conda env update -n geng -f conda.yml --prune

- conda env create -n geng -f conda.yml

- sbatch --export=ALL,POLICY_NAME=HF,TASK_FILE=configs/tasks.json ./scripts/eval_slurm.sh

- python ./scripts/run_eval.py --policy HF --task-file ./configs tasks_libero_spatial.json --n-envs 1 --episodes-per-task 2

- kaya - S00percomputer!  