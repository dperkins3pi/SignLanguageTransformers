#!/bin/bash --login

#SBATCH -C 'pascal'
#SBATCH --mem-per-cpu=10G 
#SBATCH --job-name=SignLanguage
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=3 
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4 

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