#!/bin/bash
#SBATCH --job-name=rtv_mc34_sweep
#SBATCH --partition=serial_long     # TODO: verify with `sinfo` - the MOTD mentioned a "serial_long" queue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # matches MAX_THREAD_CNT=16 used by the multiprocessing pool
#SBATCH --time=10:00:00             # TODO: adjust to the partition's max walltime
#SBATCH --output=slurm_overnight_%j.out
#SBATCH --error=slurm_overnight_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

# TODO: activate whatever environment you used for prior RollingHorizon runs, e.g.:
# module load <python module>
# source venv/bin/activate

./run_all_overnight.sh
