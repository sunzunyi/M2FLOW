from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger
import logging
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from . import flow_viz
import cv2
import logging
import time
from datetime import datetime, timedelta

def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=True, tofile=True
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + ".log")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def save_ckpt(step, scheduler, optimizer, model, args, best_val = [], latest=True):
    if latest:
        ckpt =  args.log_dir + '/ckpt' + '/latest.pth'
        state = args.log_dir + '/ckpt' + '/latest.state'
    else:
        ckpt = args.log_dir + '/ckpt' + f'/{step}_{args.name}-{best_val[0]:.3f}-best.pth'
        state = args.log_dir + '/ckpt' + f'/{step}_{args.name}-{best_val[0]:.3f}-best.state'

        
    state_dict = {
        "iter": step,
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "learning_rate": scheduler.get_last_lr()[0],
        "best_val_epe" : best_val[0],
        "best_val_step" : best_val[1],
    }
    torch.save(model.state_dict(), ckpt)
    torch.save(state_dict, state)

    ckpts = sorted(
        [x for x in os.listdir(args.log_dir + '/ckpt') if "best.pth" in x], key=lambda x: int(x.split('_')[0]), reverse=False
    )
    states = sorted(
        [x for x in os.listdir(args.log_dir + '/ckpt') if "best.state" in x], key=lambda x: int(x.split('_')[0]), reverse=False
    )
    assert len(ckpts) == len(states)
    if len(ckpts) >= 10:
        os.remove(os.path.join(args.log_dir + '/ckpt', ckpts[0]))
        os.remove(os.path.join(args.log_dir + '/ckpt', states[0]))


class Logger:
    def __init__(self, cfg, model = None, scheduler = None):
        setup_logger(
            "base",
            cfg.log_dir,
            "base_" + cfg.name,
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger_log = logging.getLogger("base")
        logger_log.info(cfg)   

        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg
        self.train_epe_list = {}
        self.train_epe_list['learing_rate'] = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

        self.logger = logger_log
        self.total_time = 0

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.cfg.sum_freq for k in sorted(self.running_loss.keys())]
        learing_rate = self.scheduler.get_last_lr()[0]
        training_str = "[{:8d}, {:8.8f}] ".format(self.total_steps, learing_rate)
        metrics_str = ', '.join(['{}:{:7.3f}'.format(k, v) for k, v in zip(sorted(self.running_loss.keys())[:-1], metrics_data[:-1])])
        # Compute time left
        time_left_sec = (self.cfg.trainer.num_steps - (self.total_steps)) * metrics_data[-1]
        time_left_sec = int(time_left_sec)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>10}"

        self.total_time = self.total_time + np.sum(self.running_loss['time'])
        time_past_sec = int(self.total_time)
        time_past_hms = "{:02d}h{:02d}m{:02d}s".format(time_past_sec // 3600, time_past_sec % 3600 // 60, time_past_sec % 3600 % 60)
        time_past_hms = f"{time_past_hms:>10}"

        self.logger.info(self.cfg.network + ' ' + training_str + '   ' + metrics_str + '   ' + time_left_hms + '  pased:' + time_past_hms)

        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.cfg.sum_freq, self.total_steps)
            if k not in self.train_epe_list.keys():
                self.train_epe_list[k] = []
            self.train_epe_list[k].append(np.mean(self.running_loss[k]))
            self.running_loss[k] = 0.0
        self.writer.add_scalar('learing_rate', learing_rate, self.total_steps)
        self.train_epe_list['learing_rate'].append(learing_rate)
        self.train_steps_list.append(self.total_steps)

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == 0:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)
            if key not in self.val_results_dict.keys():
                self.val_results_dict[key] = []
            self.val_results_dict[key].append(results[key])
        self.val_steps_list.append(self.total_steps)

        self.plot_train()
        self.plot_val()


    def close(self):
        self.writer.close()

    def plot_val(self):
        for key in self.val_results_dict.keys():
            # plot validation curve
            plt.figure()
            plt.plot(self.val_steps_list, self.val_results_dict[key])
            plt.xlabel('x_steps')
            plt.ylabel(key)
            plt.title(f'Results for {key} for the validation set')
            plt.savefig(self.cfg.log_dir+f"/{key}.png", bbox_inches='tight')
            plt.close()


    def plot_train(self):
        for key in self.train_epe_list.keys():
            # plot training curve
            plt.figure()
            plt.plot(self.train_steps_list, self.train_epe_list[key])
            plt.xlabel('x_steps')
            plt.ylabel(key)
            plt.title(f'Results for {key} , Running training')
            plt.savefig(self.cfg.log_dir+f"/train_{key}.png", bbox_inches='tight')
            plt.close()

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [""] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert len(val) == self.meters and len(n) == self.meters
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = " ".join(
            [
                "{} {:.{}f}".format(n, v, self.precision)
                for n, v in zip(self.names, self.val)
            ]
        )
        avg = " ".join(
            [
                "{} {:.{}f}".format(n, a, self.precision)
                for n, a in zip(self.names, self.avg)
            ]
        )
        return "{} ({})".format(val, avg)
