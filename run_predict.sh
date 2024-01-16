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

cd ./src/main/ &&
  python ./predict.py \
    --config_file_path ../../configs/eval_config.yaml \
    --model_type efficient_model \
    --dataset_dir ../../dataset/ \
    --submission_file_path ../../submissions/submission.csv
