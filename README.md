# PyTorch Image Classification Training Framework

Welcome to our PyTorch Image Classification Training Framework. This project provides a robust and flexible framework for training image classification models using PyTorch and Weights & Biases (wandb) for visualization.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tips](#tips)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgment](#acknowledgment)

## Project Structure

Here's a high-level overview of the project's structure:

```bash
.
├── .github                   # Directory for GitHub-specific files, used for commitlint
├── .husky                    # Directory for Husky hooks, used for commitlint
├── checkpoints               # Directory for model weights, loading points, and wandb logs
├── configs                   # Directory for configuration files
├── dataset                   # Directory for the dataset
├── submissions               # Directory for submission files
├── package-lock.json         # File used to set up the conventional commits environment
├── package.json              # File used to set up the conventional commits environment
├── run_predict.sh            # Script for making predictions on new data
├── run_split_dataset.sh      # Script for splitting the dataset into training and validation sets
├── run_train.sh              # Script for training the model on the dataset
├── LICENSE                   # MIT License file
├── README.md                 # This file, a concise description of the project
└── src                       # Source code directory
    └── main
        ├── data              # Code for loading and preprocessing the dataset
        ├── engine            # Code for defining the training and validation loops
        ├── model             # Code for defining the model architecture
        ├── options           # Code for parsing command line arguments
        ├── predict.py        # Python script for making predictions on new data
        ├── split_dataset.py  # Python script for splitting the dataset
        └── train.py          # Python script for training the model
```

## Getting Started

### Prerequisites

Ensure you have the following installed on your local machine:

- Python 3.7+
- PyTorch 1.7+
- Weights & Biases

### Installation

To install the necessary dependencies, run the following commands:

```bash
conda create -n pytorch python=3.9
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyyaml pandas tqdm wandb -c conda-forge
```

## Usage

1.Prepare Your Dataset: Place your dataset in the `./dataset` directory. The dataset should be arranged in the following format:

```bash
root/train_images/label1/xxx.png
root/train_images/label1/xxx.png
...
root/train_images/label2/xxx.png
root/train_images/label2/xxx.png
```

2.Split the Dataset: Run the `run_split_dataset.sh` script to split the dataset into training and validation sets.

3.Train the Model: Run the `run_train.sh` script to train the model on your dataset.

4.Visualize the Training Process: Log in to your Weights & Biases account to visualize the training process and performance.

## Tips

If you are using PyCharm, set `src/main` as sources root.

Set Git Bash as your default shell in Windows.

## Contributing

## License

This project is licensed under the MIT License.

## Contact

For any questions or concerns, please open an issue on GitHub.

## Acknowledgment

This project is base on:

- [pytorch/pytorch](https://github.com/pytorch/pytorch)

- [wandb/wandb](https://github.com/wandb/wandb)

- [yaml/pyyaml](https://github.com/yaml/pyyaml)

- [ahangchen/torch_base](https://github.com/ahangchen/torch_base)

- [lyhue1991/eat_pytorch_in_20_days](https://github.com/lyhue1991/eat_pytorch_in_20_days)
