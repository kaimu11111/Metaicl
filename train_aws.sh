# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ../Metaicl
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/metaicl/lib
task=threeSimTask
# python train.py \
#   --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
#   --batch_size 16 --lr 1e-2 --fp16 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
#   --num_training_steps 1000

python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
  --batch_size 1 --lr 1e-2 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
  --num_training_steps 1000