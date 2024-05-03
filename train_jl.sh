#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=metaicl
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4

source activate metaicl
nvidia-smi

cd ../Metaicl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mhong/li003755/.conda/envs/metaicl/lib/
task=threeSimTask
python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
  --batch_size 16 --lr 1e-5 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
  --num_training_steps 10000

# task=threeDiffTask
# python train.py \
#   --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
#   --batch_size 16 --lr 1e-5 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
#   --num_training_steps 10000

# --fp16
exit