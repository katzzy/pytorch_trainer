import os
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
from data.augment import transform_train
from data.augment import transform_eval


def select_train_loader(args):
    train_dataset_dir = os.path.join(args.dataset_dir, 'train_valid_test', "train")
    train_dataset_dir = Path(train_dataset_dir).as_posix()
    train_dataset = torchvision.datasets.ImageFolder(train_dataset_dir, transform=transform_train)
    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=False)
    return train_loader


def select_eval_loader(args):
    eval_dataset_dir = os.path.join(args.dataset_dir, 'train_valid_test', "valid")
    eval_dataset_dir = Path(eval_dataset_dir).as_posix()
    eval_dataset = torchvision.datasets.ImageFolder(eval_dataset_dir, transform=transform_eval)
    val_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    return val_loader
