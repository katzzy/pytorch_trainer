#!/bin/bash
uNames=$(uname -s);
osName=${uNames:0:10};
if [ "$osName" = "Darwin" ]; then
  echo "Current system is macOS!";
  conda activate pytorch;
elif [ "$osName" = "Linux" ]; then
  echo "Current system is Linux!";
  conda activate pytorch;
elif [ "$osName" = "MINGW64_NT" ]; then
  echo "Current system is Windows!";
  conda activate pytorch;
else
  echo "Current system is not supported!";
  exit;
fi

cd ./src/main/ && \
python ./split_dataset.py \
--config_file_path ../../configs/split_config.yaml
#--dataset_dir ../../dataset/paddy-disease-classification/