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
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../MetaICL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mhong/wan00559/.conda/envs/Metaicl/lib/
task=threeSimTask
python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
  --batch_size 16 --lr 1e-2 --fp16 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
  --num_training_steps 1000
