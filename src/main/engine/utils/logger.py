import os
import torch
import wandb
from pathlib import Path


class Recoder:
    def __init__(self):
        self.__metrics = {}

    def record(self, name, value):
        if name in self.__metrics.keys():
            if torch.is_tensor(value):
                self.__metrics[name].append(value.item())
            else:
                self.__metrics[name].append(value)
        else:
            if torch.is_tensor(value):
                self.__metrics[name] = [value.item()]
            else:
                self.__metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.__metrics.keys():
            kvs[key] = sum(self.__metrics[key]) / len(self.__metrics[key])
        return kvs

    def clear_metrics(self):
        for key in self.__metrics.keys():
            del self.__metrics[key][:]
            self.__metrics[key] = []


class Logger:
    def __init__(self, args):
        logs_path = os.path.join(args.checkpoints_dir, 'logs')
        logs_path = Path(logs_path).as_posix()
        if not os.path.exists(logs_path):
            os.makedirs(logs_path, exist_ok=True)
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="paddy-doctor",

            dir=logs_path,
            # track hyperparameters and run metadata
            config=args.__dict__
        )
        self.__recoder = Recoder()
        self.__checkpoints_dir = args.checkpoints_dir
        self.__gpus = args.gpus
        self.best_acc = 0
        self.best_acc_epoch = 0

    def set_best_score(self, best_acc, best_acc_epoch):
        self.best_acc = best_acc
        self.best_acc_epoch = best_acc_epoch

    @staticmethod
    def __tensor2img(tensor):
        # implement according to your data
        return tensor.cpu().detach().numpy()

    def record_scalar(self, name, value):
        self.__recoder.record(name, value)

    def clear_scalar_cache(self):
        self.__recoder.clear_metrics()

    def save_curves(self, epoch):
        pass

    def save_imgs(self, names2imgs, epoch):
        pass

    @staticmethod
    def finish_wandb():
        wandb.finish()

    def print_logs(self, epoch, execution_time):
        print('Summary:')
        kvs = self.__recoder.summary()
        for key in kvs.keys():
            wandb.log({key: kvs[key]}, step=epoch)
            print(key + ' = {}'.format(kvs[key]))
        print('Execution time(in secs) = {}'.format(execution_time))
        [self.best_acc, self.best_acc_epoch] = [kvs['val/accuracy'], epoch] \
            if kvs['val/accuracy'] > self.best_acc else [self.best_acc, self.best_acc_epoch]
        wandb.log({'best_acc': self.best_acc}, step=epoch)
        print('Best accuracy = {} in epoch = {}'.format(self.best_acc, self.best_acc_epoch))

    def save_checkpoint(self, epoch, model, optimizer, scheduler):
        weights_name = 'weights_{epoch:03d}.pth'.format(epoch=epoch)
        checkpoint_name = 'checkpoint_{epoch:03d}.tar'.format(epoch=epoch)
        weights_files_dir = os.path.join(self.__checkpoints_dir, 'weights_files')
        weights_files_dir = Path(weights_files_dir).as_posix()
        if not os.path.exists(weights_files_dir):
            os.makedirs(weights_files_dir, exist_ok=True)
        checkpoint_files_dir = os.path.join(self.__checkpoints_dir, 'checkpoint_files')
        checkpoint_files_dir = Path(checkpoint_files_dir).as_posix()
        if not os.path.exists(checkpoint_files_dir):
            os.makedirs(checkpoint_files_dir, exist_ok=True)
        weights_path = os.path.join(weights_files_dir, weights_name)
        weights_path = Path(weights_path).as_posix()
        checkpoint_path = os.path.join(checkpoint_files_dir, checkpoint_name)
        checkpoint_path = Path(checkpoint_path).as_posix()
        if len(self.__gpus) == 0 or len(self.__gpus) == 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': self.best_acc,
                'best_acc_epoch': self.best_acc_epoch,
            }
            torch.save(model.state_dict(), weights_path)
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model.module.state_dict(), weights_path)
            torch.save(checkpoint, checkpoint_path)
