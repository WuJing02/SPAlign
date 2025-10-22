#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu "3 gmem=30000"
#JSUB -n 4
#JSUB -app default
#JSUB -e output/error.%J
#JSUB -o output/output.%J
#JSUB -J gps
#JSUB -I

source /apps/software/anaconda3/etc/profile.d/conda.sh
module load cuda/11.8
conda activate qwen2.5
python run_tuning.py

