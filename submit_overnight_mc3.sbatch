#!/bin/bash
#SBATCH --job-name=rtv_mc3_sweep
#SBATCH --partition=teramem_inter   # 10-day walltime cap, safest for an unattended overnight run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # matches MAX_THREAD_CNT=16 used by the multiprocessing pool
#SBATCH --mem=32G                   # modest slice of the node's 6TB, no need for more
#SBATCH --time=12:00:00             # generous overnight window for just the 4 mc=3 scripts
#SBATCH --output=slurm_overnight_mc3_%j.out
#SBATCH --error=slurm_overnight_mc3_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

# TODO: activate whatever environment you used for prior RollingHorizon runs, e.g.:
# module load <python module>
# source venv/bin/activate

./run_all_overnight_mc3.sh
