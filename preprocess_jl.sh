#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.out
#SBATCH --job-name=metaicl
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4

source activate metaicl
cd ../Metaicl

cd preprocess
# preprocess from crossfit
CUDA_LAUNCH_BLOCKING=1 python _build_gym.py --build --n_proc=40 --do_test
CUDA_LAUNCH_BLOCKING=1 python _build_gym.py --build --n_proc=40 --do_train # skip if you won't run training yourself
# preprocess from unifiedqa
# CUDA_LAUNCH_BLOCKING=1 python unifiedqa.py --do_train --do_test # skip `--do_train` if you won't run training yourself

# CUDA_LAUNCH_BLOCKING=1