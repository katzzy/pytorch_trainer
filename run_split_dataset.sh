#!/bin/bash
uNames=$(uname -s);
osName=${uNames:0:10};
if [ "$osName" = "Darwin" ]; then
  echo "Current system is macOS!";
  source activate pytorch;
elif [ "$osName" = "Linux" ]; then
  echo "Current system is Linux!";
  source /home/korbin/miniconda3/bin/activate pytorch;
elif [ "$osName" = "MINGW64_NT" ]; then
  echo "Current system is Windows!";
  source D:/miniconda3/Scripts/activate pytorch;
else
  echo "Current system is not supported!";
  exit;
fi

cd ./src/main/ && \
python ./split_dataset.py \
--config_file_path ../../configs/split_config.yaml
#--dataset_dir ../../dataset/paddy-disease-classification/