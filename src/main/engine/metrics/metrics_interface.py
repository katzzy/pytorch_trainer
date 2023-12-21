import torch
from engine.metrics.loss.focal_loss import FocalLoss


def select_loss(args):
    type2lossfunction = {
        'focal_loss': FocalLoss(num_class=10),
    }
    loss_function = type2lossfunction[args.loss_type]
    return loss_function


def select_acc(args):
    type2accfunction = {
        'classification_acc': classification_accuracy,
    }
    acc_function = type2accfunction[args.acc_type]
    return acc_function


def classification_accuracy(output_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
    num_acc_label = (output_batch.argmax(dim=1) == label_batch).float().sum().item()
    num_total_label = label_batch.shape[0]
    acc = num_acc_label / num_total_label
    return acc
