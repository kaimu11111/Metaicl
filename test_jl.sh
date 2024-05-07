#!/bin/bash

#SBATCH --time=2:00:00
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

# echo "Test on threeDiffTask"
# task=threeDiffTask
# python test.py \
#   --dataset ethos-directed_vs_generalized --k 16 --split test --seed 100 --use_demonstrations \
#   --test_batch_size 16 --method channel --out_dir checkpoints/channel-metaicl/threeDiffTask --global_step 10000

# task=rotten_tomatoes
# python train.py \
#   --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method channel \
#   --do_tensorize --n_gpu 1 --n_process 40

echo "Test on oneSimTask"
task=oneSimTask
python test.py \
  --dataset rotten_tomatoes --k 16 --split test --seed 100 --use_demonstrations \
  --test_batch_size 16 --method channel --out_dir checkpoints/channel-metaicl/oneSimTask --global_step 5000

echo "Test on oneDiffTask"
task=oneDiffTask
python test.py \
  --dataset rotten_tomatoes --k 16 --split test --seed 100 --use_demonstrations \
  --test_batch_size 16 --method channel --out_dir checkpoints/channel-metaicl/oneDiffTask --global_step 5000

# --fp16
exit