#!/bin/bash -l
#SBATCH --job-name=1b_
#SBATCH --cpus-per-task=24
#SBATCH --output=output/sbatch_output/a_05.txt
#SBATCH --gres=gpu:3
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=2-00:00:00
#SBATCH --exclude=c[1-16]
# #SBATCH --exclude=c[11,17,18,19,20]
# #SBATCH --nodelist=c[9,10,12,13,17,18,19,20]

module --silent anaconda3/2023.09-0-gcc-13.2.0-6gotrib
conda activate gofaster00

echo "GPU CHECK"
nvidia-smi

echo "==============================================================================================================="
echo "PYTHON SCRIPT:"

accelerate launch code/1_mymethod_embed.py

echo "python path executed"

echo "SBATCH FINISHED"
