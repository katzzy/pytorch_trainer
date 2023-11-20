## Project structure

```
.
├── checkpoints
├── configs
├── dataset
├── submissions
├── run_predict.sh
├── run_split_dataset.sh
├── run_tensorboard.sh
├── run_train.sh
├── LICENSE
├── README.md
└── src
    └── main
        ├── data
        ├── engine
        ├── model
        ├── options
        ├── predict.py
        ├── split_dataset.py
        └── train.py
```

## Tips

If you are using pycharm, set `src/main` as sources root.

Set git bash as your default shell in windows.

## How to use

### 1, Install requirements

`pip install tensorboard`

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch pyyaml pandas tqdm`

### 2, Put your dataset to `./dataset`

A generic dataset folder where the images are arranged in this way by default: 
```
root/train_images/label1/xxx.png
root/train_images/label1/xxx.png
...
root/train_images/label2/xxx.png
root/train_images/label2/xxx.png
...
...
root/test_images/xxy.png
root/test_images/xxy.png
...
```

### 3, Set your parameter

You can set parameter in `config/xxx.yaml` which will be set as default parameter.

Then, you can override all parameter in shell script.

### 4, Split dataset

Running Script `./run_split_dataset.sh`

### 5, Create model

Create your model in `src/main/model`

### 6, Train

Running Script `./run_train.sh`

### 7, Predict

Running Script `./run_predict.sh`

Every row of submission file should already have an ID

## Acknowledgment

This project is base on 

* [ahangchen/torch_base](https://github.com/ahangchen/torch_base)

* [lyhue1991/eat_pytorch_in_20_days](https://github.com/lyhue1991/eat_pytorch_in_20_days)

* [pytorch/pytorch](https://github.com/pytorch/pytorch)

* [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard)

* [yaml/pyyaml](https://github.com/yaml/pyyaml)
