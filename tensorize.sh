#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.out
#SBATCH --job-name=threeSimTask
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong
task=threeSimTask
python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method channel \
  --do_tensorize --n_gpu 8 --n_process 40