#!/bin/bash --login

#SBATCH --partition=m13h           # or m13l, m9g, m8g as available
#SBATCH --job-name=SignLanguage
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4               # set to the max per node (or less)
#SBATCH --cpus-per-task=8          # adjust for your workload
#SBATCH --mem=120G                 # set less than total per node
#SBATCH --ntasks-per-node=4        # 1 per GPU for DDP

module load cuda/12.4
# Print out useful data
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK, SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

nvidia-smi
nvcc --version

cd $HOME/SignLanguageTransformers/
mamba run -n SignLanguage python training.py