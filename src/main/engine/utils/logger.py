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
        self.__recoder = Recoder()
        self.__kvs = {}
        self.__checkpoints_dir = args.checkpoints_dir
        self.__gpus = args.gpus
        self.__is_wandb_on = args.is_wandb_on
        self.best_acc = 0
        self.best_acc_epoch = 0
        logs_path = os.path.join(args.checkpoints_dir, 'logs')
        logs_path = Path(logs_path).as_posix()
        if not os.path.exists(logs_path):
            os.makedirs(logs_path, exist_ok=True)
        if self.__is_wandb_on:
            wandb.init(
                project="paddy-doctor",
                job_type="training",
                tags=args.tags,
                dir=logs_path,
                config=args.__dict__
            )
        else:
            pass

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

    def summary(self, epoch):
        self.__kvs = self.__recoder.summary()
        [self.best_acc, self.best_acc_epoch] = [self.__kvs['val/accuracy'], epoch] \
            if self.__kvs['val/accuracy'] > self.best_acc else [self.best_acc, self.best_acc_epoch]

    def save_curves(self, epoch):
        if self.__is_wandb_on:
            for key in self.__kvs.keys():
                wandb.log({key: self.__kvs[key]}, step=epoch)
            wandb.log({'best_accuracy': self.best_acc}, step=epoch)
        else:
            pass

    def save_imgs(self, names2imgs, epoch):
        pass

    def finish_wandb(self):
        if self.__is_wandb_on:
            wandb.finish()
        else:
            pass

    def print_logs(self, execution_time):
        print('Summary:')
        for key in self.__kvs.keys():
            print('{} = {}'.format(key, self.__kvs[key]))
        print('Execution Time: {}s'.format(execution_time))
        print('Best Accuracy: {} in epoch {}'.format(self.best_acc, self.best_acc_epoch))

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
