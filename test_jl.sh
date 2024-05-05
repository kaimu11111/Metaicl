#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=metaicl
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong

source activate metaicl
nvidia-smi

cd ../Metaicl
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mhong/li003755/.conda/envs/metaicl/lib/
# echo "Test on threeSimTask"
# task=threeSimTask
# python test.py \
#   --dataset ethos-directed_vs_generalized --k 16 --split test --seed 100 --use_demonstrations \
#   --test_batch_size 16 --method channel --out_dir checkpoints/channel-metaicl/threeSimTask --global_step 10000

echo "Test on threeDiffTask"
task=threeDiffTask
python test.py \
  --dataset ethos-directed_vs_generalized --k 16 --split test --seed 100 --use_demonstrations \
  --test_batch_size 16 --method channel --out_dir checkpoints/direct-metaicl/threeSimTask --global_step 10000

# --fp16
exit