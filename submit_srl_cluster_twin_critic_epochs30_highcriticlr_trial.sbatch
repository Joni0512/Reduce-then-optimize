#!/bin/bash
#SBATCH --job-name=srl_twin_hclr
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_twin_hclr_%j.log
#SBATCH --error=srl_twin_hclr_%j.err

# Twin-critic, EPOCHS=30, critic_lr raised to 0.005 (actor_lr unchanged) - see chat, testing
# whether pushing critic_lr further helps beyond the best local run so far (best_val=0.7085
# at critic_lr=0.00108). SEED exported by the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - submit with: sbatch --export=SEED=<n> submit_srl_cluster_twin_critic_epochs30_highcriticlr_trial.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_epochs30_highcriticlr_trial "$SEED"
