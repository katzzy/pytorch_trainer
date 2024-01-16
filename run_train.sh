#!/bin/bash
uNames=$(uname -s)
osName=${uNames:0:10}
if [ "$osName" = "Darwin" ]; then
  echo "Current system is macOS!"
  source /Users/korbin/anaconda3/bin/activate pytorch
elif [ "$osName" = "Linux" ]; then
  echo "Current system is Linux!"
  source /home/korbin/miniconda3/bin/activate pytorch
elif [ "$osName" = "MINGW64_NT" ]; then
  echo "Current system is Windows!"
  source "D:\miniconda3\Scripts\activate.bat" pytorch
else
  echo "Other system: $osName"
  exit
fi

if [ ! -d "checkpoints" ]; then
  mkdir checkpoints
fi
cd ./src/main/ &&
  python ./train.py \
    --config_file_path ../../configs/train_config.yaml \
    --epochs 500 \
    --batch_size 32 \
    --dataset_dir ../../dataset/paddy-disease-classification/ \
    --model_type efficient_model |
  tee ../../checkpoints/output.txt
