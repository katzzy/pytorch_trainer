import os
import sys
import datetime
import torch
import torch.optim
import torch.utils.data
import numpy as np
import random
from tqdm import tqdm
from engine.utils.logger import Logger
from engine.metrics.metrics_interface import select_loss
from engine.metrics.metrics_interface import select_acc


class Trainer(object):
    def __init__(self, args, model, train_loader, val_loader):
        self.__setup_seed(args.seed)
        self.__logger = Logger(args)
        self.__loss_function = select_loss(args)
        self.__acc_function = select_acc(args)
        self.__train_loader = train_loader
        self.__val_loader = val_loader
        self.__start_epoch = 1
        self.__end_epoch = args.epochs

        train_mode = 'Train'
        train_status_logs = []

        # loading model
        self.__model = model
        if args.pretrained_weights_path is not None:
            train_mode = 'Finetune'
            self.__model.load_state_dict(torch.load(args.pretrained_weights_path), strict=args.is_load_strict)
            train_status_logs.append('Load pretrained weights from {}'.format(args.pretrained_weights_path))

        if args.checkpoint_path is not None:
            train_mode = 'Resume'
            checkpoint = torch.load(args.checkpoint_path)
            self.__start_epoch = checkpoint['epoch'] + 1
            self.__model.load_state_dict(checkpoint['model_state_dict'], strict=args.is_load_strict)
            self.__logger.set_best_score(checkpoint['best_acc'], checkpoint['best_acc_epoch'])
            train_status_logs.append('Load checkpoint from {}'.format(args.checkpoint_path))
            train_status_logs.append('Load best score: {0:.4f} at epoch {1}'.format(checkpoint['best_acc'],
                                                                                    checkpoint['best_acc_epoch']))

        if len(args.gpus) == 0:
            gpu_mode = 'CPU'
            self.device = torch.device('cpu')
            pass
        elif len(args.gpus) == 1:
            gpu_mode = 'Single-GPU'
            self.device = torch.device('cuda:{}'.format(args.gpus[0]))
            self.__model.to(self.device)
        else:
            gpu_mode = 'Multi-GPU'
            self.__model = torch.nn.DataParallel(self.__model, device_ids=args.gpus)

        # initialize the optimizer
        self.__optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.__model.parameters()),
                                            args.lr, betas=(args.momentum, args.beta),
                                            weight_decay=args.weight_decay)

        self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, step_size=30, gamma=0.1)
        if args.checkpoint_path is not None:
            checkpoint = torch.load(args.checkpoint_path)
            self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.__scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            train_status_logs.append('Load optimizer from {}'.format(args.checkpoint_path))
            train_status_logs.append('Load scheduler from {}'.format(args.checkpoint_path))

        # print status
        print('==========' * 10)
        print('Model:')
        print(self.__model)
        print('==========' * 10)
        print('Trainable Parameters:')
        for name, param in self.__model.named_parameters():
            if param.requires_grad:
                print(name)
        print('==========' * 10)
        print('Wandb Mode: {}'.format(args.is_wandb_on))
        print('Train Mode: {}'.format(train_mode))
        print('GPU Mode: {}'.format(gpu_mode))
        print('==========' * 10)
        for log in train_status_logs:
            print(log)
        print('==========' * 10)

    def train(self):
        try:
            print('start training...')
            for epoch in range(self.__start_epoch, self.__end_epoch + 1):
                # train for one epoch
                since_time = datetime.datetime.now()
                self.__print_now_time('Epoch {0} / {1}'.format(epoch, self.__end_epoch))
                self.__train_per_epoch()
                self.__val_per_epoch()
                self.__logger.save_checkpoint(epoch, self.__model, self.__optimizer, self.__scheduler)
                self.__logger.summary(epoch)
                self.__logger.save_curves(epoch)
                self.__logger.print_logs((datetime.datetime.now() - since_time).seconds)
                self.__logger.clear_scalar_cache()
        except KeyboardInterrupt:
            print('=> Interrupted Training!')
            self.__logger.finish_wandb()
        else:
            print('=> Finished Training!')
            self.__logger.finish_wandb()

    def __train_per_epoch(self):
        print('Train:')
        # switch to train mode
        self.__model.train()
        loop = tqdm(enumerate(self.__train_loader), total=len(self.__train_loader), file=sys.stdout)
        for i, data_batch in loop:
            input_batch, output_batch, label_batch = self.__step(data_batch)

            # compute loss and acc
            loss, metrics = self.__compute_metrics(output_batch, label_batch, is_train=True)

            # compute gradient and do Adam step
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            # logger record
            for key in metrics.keys():
                self.__logger.record_scalar(key, metrics[key])
            loop.set_postfix(**metrics)
        loop.close()
        self.__scheduler.step()

    @torch.no_grad()
    def __val_per_epoch(self):
        print('Eval:')
        # switch to eval mode
        self.__model.eval()

        with torch.no_grad():
            loop = tqdm(enumerate(self.__val_loader), total=len(self.__val_loader), file=sys.stdout)
            for i, data_batch in loop:
                input_batch, output_batch, label_batch = self.__step(data_batch)

                # compute loss and acc
                loss, metrics = self.__compute_metrics(output_batch, label_batch, is_train=False)

                for key in metrics.keys():
                    self.__logger.record_scalar(key, metrics[key])
                loop.set_postfix(**metrics)
            loop.close()

    def __step(self, data_batch):
        input_batch, label_batch = data_batch
        # warp input
        input_batch = input_batch.to(self.device)
        label_batch = label_batch.to(self.device)

        # compute output
        output_batch = self.__model(input_batch)
        return input_batch, output_batch, label_batch

    def __compute_metrics(self, output_batch, label_batch, is_train):
        # you can call functions in metrics_interface.py
        loss = self.__calculate_loss(output_batch, label_batch)
        acc = self.__evaluate_accuracy(output_batch, label_batch)
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'loss': loss.item(),
            prefix + 'accuracy': acc,
        }
        return loss, metrics

    def __calculate_loss(self, output_batch: torch.Tensor, label_batch: torch.Tensor) -> torch.Tensor:
        loss = self.__loss_function(output_batch, label_batch)
        return loss

    def __evaluate_accuracy(self, output_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
        acc = self.__acc_function(output_batch, label_batch)
        return acc

    @staticmethod
    def __print_now_time(info):
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('==========' * 10)
        print('[{0}] {1}'.format(now_time, info))

    @staticmethod
    def __setup_seed(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def __gen_img_to_write(img, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': img[0],
        }
