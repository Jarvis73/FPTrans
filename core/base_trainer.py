import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from constants import on_cloud
from core import losses as loss_utils
from core import solver
from core.metrics import FewShotMetric, Accumulator
from data_kits import datasets
from utils_.loggers import C as CC
from utils_.timer import Timer


def round_(array):
    if isinstance(array, float) or array.ndim == 0:
        return f"{array:5.2f}"
    if array.ndim == 1:
        return "[" + ", ".join([f"{x:5.2f}" for x in array]) + "]"


def save_img(file_path, img):
    pil_img = Image.fromarray(img.astype(np.uint8))
    # pil_img.putpalette(palette)
    pil_img.save(file_path)


def pad(str0, padded, length, align='left'):
    remains = length - len(str0)
    if align == 'left':
        left = 1
    elif align == 'center':
        left = remains // 2
    else:
        raise ValueError
    right = remains - left
    return padded * left + str0 + padded * right


class BaseEvaluator(object):
    """
    Evaluator base class. Evaluator is used in the validation stage and testing stage.
    All the evaluators should inherit from this class and implement the `test_step()`
    function.

    Parameters
    ----------
    opt: misc.MapConfig
        Experiment configuration.
    model: nn.Module
        PyTorch model instance.
    mode: str
        Evaluation mode. [EVAL_ONLINE, EVAL]

    """
    def __init__(self, opt, logger, device, model, model_T, mode, need_grad=False):
        self.opt = opt
        self.logger = logger
        self.device = device
        self.mode = mode
        if mode not in ["EVAL_ONLINE", "EVAL"]:
            raise ValueError(f"Not supported evaluation mode {mode}. [EVAL_ONLINE, EVAL]")
        self.num_devices = torch.cuda.device_count()

        self.model = model
        if not isinstance(self.model, nn.DataParallel):
            self.model_DP = self.init_device(self.model)
        else:
            self.model_DP = self.model
        # if model_T is not None:
        #     self.model_T = model_T
        #     if not isinstance(self.model_T, nn.DataParallel):
        #         self.model_T_DP = self.init_device(self.model_T)
        #     else:
        #         self.model_T_DP = self.model_T
        # else:
        #     self.model_T = None
        #     self.model_T_DP = None
        
        self.loss_obj = loss_utils.get(opt, logger, loss=opt.loss.replace('dt', ''))
        self.need_grad = need_grad

    def test_step(self, batch, step):
        raise NotImplementedError

    def init_device(self, net):
        net = net.to(self.device)
        net_DP = nn.DataParallel(net, device_ids=range(self.num_devices))
        return net_DP

    def maybe_print_metrics(self, accu, gen, epoch=None):
        if self.opt.tqdm:
            if epoch is not None:
                print_str = f"[{self.mode}] [{epoch}/{self.opt.epochs}] loss: {accu.mean('loss'):.5f}"
            else:
                print_str = f"[{self.mode}] loss: {accu.mean('loss'):.5f}"
            gen.set_description(print_str)

    @torch.no_grad()
    def start_eval_loop(self, data_loader, num_classes, epoch=None):
        """ For few-shot learning """
        # Set model to evaluation mode (for specific layers, such as batchnorm, dropout, dropblock)
        self.model.eval()
        # Fix sampling order of the test set.
        data_loader.dataset.reset_sampler()
        timer = Timer()
        data_timer = Timer()
        val_labels = datasets.get_val_labels(self.opt, self.mode)
        # create saving directory
        save_dir = None
        if self.opt.save_dir is not None:
            save_dir = Path(self.opt.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        fs_metric = FewShotMetric(num_classes)    
        accu = Accumulator(loss=0)
        data_loader.dataset.sample_tasks()

        gen = data_loader
        if self.opt.tqdm:
            gen = tqdm(gen, leave=True)
        data_timer.tic()
        for i, batch in enumerate(gen, start=1):
            # TODO: remove
            if i < self.opt.test_start:
                continue
            data_timer.toc()
            with timer.start():
                qry_pred, losses = self.test_step(batch, i)
                accu.update(**losses)
            self.maybe_print_metrics(accu, gen, epoch)
            fs_metric.update(qry_pred, batch['qry_msk'], batch['cls'])

            # save prediction
            if save_dir:
                qry_mask = batch['qry_msk'].cpu().numpy()[:, 0]
                qry_pred = qry_pred[:, 0]
                classes = batch['cls'].cpu().numpy().tolist()
                for j, qry_name in enumerate(batch['qry_names'][0]):
                    if batch['qry_ori_size'] is not None:
                        ori_H, ori_W = batch['qry_ori_size'][j]
                        p = qry_pred[j, :ori_H, :ori_W] * 255
                        r = (qry_mask[j, :ori_H, :ori_W] == 1) * 255
                        save_path = save_dir / (f'{(i - 1) * self.opt.test_bs + j + 1:04d}_' + qry_name + '_pred.png')
                        save_img(save_path, p)
                        save_path = save_dir / (f'{classes[j]:02d}_' + qry_name + '_mask.png')
                        save_img(save_path, r)
            data_timer.tic()

        miou_class, miou_avg = fs_metric.get_scores(val_labels)
        biou_class, biou_avg = fs_metric.get_scores(val_labels, binary=True)
        str1 = f'mIoU mean: {round_(miou_class * 100)} ==> {round_(miou_avg * 100)}'
        str2 = f'bIoU mean: {round_(biou_class * 100)} ==> {round_(biou_avg * 100)}'

        if self.mode == "EVAL_ONLINE":
            self.logger.info(str1)
            self.logger.info(str2)
        elif self.mode == "EVAL":
            str3 = f'speed: {round_(timer.cps)} FPS'
            max_length = max(len(str1), len(str2), len(str3)) + 2
            self.logger.info('╒' + pad(' Final Results ', '═', max_length, align='center') + '╕')
            self.logger.info('│' + pad(str1, ' ', max_length) + '│')
            self.logger.info('│' + pad(str2, ' ', max_length) + '│')
            self.logger.info('│' + pad(str3, ' ', max_length) + '│')
            self.logger.info('╘' + pad('', '═', max_length) + '╛')

        return accu.mean('loss'), miou_avg, biou_avg, timer.elapsed, data_timer.elapsed


class BaseTrainer(object):
    def __init__(self, opt, logger, device, model, data_loader, data_loader_val, _run):
        self.opt = opt
        self.logger = logger
        self.device = device
        self.run = _run
        self.data_loader = data_loader
        self.data_loader_val = data_loader_val
        self.num_devices = torch.cuda.device_count()

        # Define model-related objects
        self.model = model
        self.model_DP = self.init_device(self.model)
        self.loss_obj = loss_utils.get(opt, logger)

        self.build_optimizer(verbose=1)
        self.step_lr_counter = 0

        # Define model_dir for saving checkpoints
        self.do_ckpt = True
        self.log_dir = _run.run_dir
        if on_cloud:
            self.cloud_save_dir = Path(f"afs/output/models_fss/{self.log_dir.name}")
            if not self.cloud_save_dir.exists():
                self.cloud_save_dir.mkdir(parents=True)

        # Define metrics and output templates
        self.best_iou = -1.
        self.template = "[{:d}/{:d}]" + \
                        " | Tr {:6.4f} | Val {:6.4f} | mIoU {:5.2f} | bIoU {:5.2f}" + \
                        " | DATA {:s} | OPT {:s} | ETA {:s}"

        self.loss_names = 'loss prompt pair'.split()
        self.steps_per_epoch = len(self.data_loader)

    def build_optimizer(self, verbose=0):
        opt = self.opt
        model = self.model

        param_list = model.get_params_list()
        max_steps = opt.epochs * len(self.data_loader)
        self.optimizer, self.scheduler = solver.get(opt, param_list, max_steps)

        if verbose:
            plists = model.get_params_list()
            for plist in plists:
                self.logger.info(f"Number of trainable parameters: {len([_ for _ in plist['params']])}")

    def train_step(self, batch, step, epoch):
        raise NotImplementedError

    def init_device(self, net):
        net = net.to(self.device)
        net_DP = nn.DataParallel(net, device_ids=range(self.num_devices))
        return net_DP

    @staticmethod
    def second2str(seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds_ = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_:02d}"

    def time_left(self, iters, epoch, speed, extra_per_epoch=0):
        iters_left = (self.opt.epochs - epoch) * self.steps_per_epoch + (self.steps_per_epoch - iters)
        time_left = int(iters_left * speed) + (self.opt.epochs - epoch) * extra_per_epoch
        return time_left

    def maybe_print_losses(self, iters, epoch, accu, timer, gen, data_timer):
        if self.opt.tqdm:
            fmt_str = "[TRAIN] [{:d}/{:d}] "
            for name in self.loss_names:
                fmt_str += name + ": {:.4f} "
            print_str = fmt_str.format(epoch, self.opt.epochs, *accu.mean(self.loss_names))
            gen.set_description(print_str)
        elif iters % self.opt.print_interval == 0 or iters == self.steps_per_epoch:
            fmt_str = "[{:d}/{:d}] [{:d}/{:d}] "
            for name in self.loss_names:
                fmt_str += name + ": {:.4f} "
            fmt_str += "[SPD: {:.2f}]"
            print_str = fmt_str.format(
                epoch, self.opt.epochs, 
                iters, self.steps_per_epoch, 
                *accu.mean(self.loss_names), 
                timer.spc + data_timer.spc,
            )
            self.logger.info(print_str)

    def print_lr(self):
        for i, group in enumerate(self.optimizer.param_groups):
            self.logger.info(f"Learning rate for {self.opt.optim.upper()} param_group[{i}] is {group['lr']:.6g}")
        if hasattr(self, 'Toptimizer'):
            for i, group in enumerate(self.Toptimizer.param_groups):
                self.logger.info(f"Learning rate for {self.opt.Toptim.upper()} param_group[{i}] is {group['lr']:.6g}")

    def start_training_loop(self, start_epoch, evaluator, num_classes):
        timer = Timer()
        data_timer = Timer()
        accu = Accumulator(**{x: 0. for x in self.loss_names})

        for epoch in range(start_epoch, self.opt.epochs + 1):
            self.print_lr()

            # 1. Training
            self.model.train()
            self.data_loader.dataset.sample_tasks()
            gen = self.data_loader
            if self.opt.tqdm:
                gen = tqdm(gen, leave=True)

            for i, batch in enumerate(gen, start=1):
                if i > 1:
                    data_timer.toc()
                with timer.start():
                    losses = self.train_step(batch, i, epoch)
                    accu.update(**losses)
                self.maybe_print_losses(i, epoch, accu, timer, gen, data_timer)
                self.step_lr()
                data_timer.tic()

            # 2. Evaluation
            if self.opt.ckpt_interval > 0 and epoch % self.opt.ckpt_interval == 0 or epoch == self.opt.epochs:
                mloss, miou, biou, val_elapsed, val_data_elapsed = evaluator.start_eval_loop(self.data_loader_val, num_classes, epoch)
                best = self.snapshot(epoch, miou, biou)
                
                data_time = data_timer.elapsed + val_data_elapsed
                epoch_time = timer.elapsed + val_elapsed
                self.log_result(epoch, accu, mloss, miou, biou, best, epoch_time, data_time)

            # 3. Prepare for next epoch
            timer.reset()
            data_timer.reset()
            accu.reset()

    def step_lr(self):
        """
        Update learning rate by the specified learning rate policy.
        For 'cosine' and 'poly' policies, the learning rate is updated by steps.
        For other policies, the learning rate is updated by epochs.
        """
        if self.scheduler is None:
            return

        self.step_lr_counter += 1

        if self.opt.lrp in ["cosine", "poly", "cosinev2"]:   # forward per step
            self.scheduler.step()
        elif self.step_lr_counter == self.steps_per_epoch:      # forward per epoch
            self.scheduler.step()
            self.step_lr_counter = 0

    def get_weights(self):
        state_dict = self.model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if 'original_encoder' not in k}
        return state_dict

    def snapshot(self, epoch, miou, biou, verbose=0):
        best = False
        if on_cloud and self.log_dir.stem == "None":
            if miou > self.best_iou:
                best = True
                self.best_iou = miou
            return best

        if miou > self.best_iou:
            best = True
            self.best_iou = miou

        save_path = self.log_dir / "ckpt.pth"
        state = {
            'epoch': epoch,
            'miou': miou,
            'biou': biou,
            'best_miou': self.best_iou,
            'model_state': self.get_weights(),
        }

        torch.save(state, save_path)
        if verbose:
            self.logger.info(CC.c(f" \\_/ Save checkpoint to {save_path}", CC.OKGREEN))

        if best:
            shutil.copyfile(save_path, self.log_dir / "bestckpt.pth")
        
        # Make a copy when oncloud
        if on_cloud:
            state.update({
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state().numpy(),
                'train_sampler_state': self.data_loader.dataset.sampler.get_state(),
                'val_sampler_state': self.data_loader_val.dataset.sampler.get_state(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
            })
            if hasattr(self, 'Toptimizer'):
                state['Toptimizer_state'] = self.Toptimizer.state_dict()

            try:
                on_cloud_save_path = self.cloud_save_dir / "ckpt.pth"
                if on_cloud_save_path.exists():
                    os.remove(on_cloud_save_path)
                torch.save(state, on_cloud_save_path)
            except Exception as e:
                print(e)
            
            if best:
                try:
                    on_cloud_save_path_best = self.cloud_save_dir / "bestckpt.pth"
                    if on_cloud_save_path_best.exists():
                        os.remove(on_cloud_save_path_best)
                    shutil.copyfile(save_path, on_cloud_save_path_best)
                except Exception as e:
                    print(e)
        return best

    def log_result(self, epoch, accu, val_loss, val_mIoU, val_bIoU, best, epoch_time, data_time, **kwargs):
        # Log epoch summary to the terminal
        losses = accu.mean('loss')
        log_str = self.template.format(
            epoch, 
            self.opt.epochs, 
            losses, 
            val_loss, 
            val_mIoU * 100, 
            val_bIoU * 100, 
            self.second2str(data_time),
            self.second2str(epoch_time),
            self.second2str((self.opt.epochs - epoch) * epoch_time)
        ) + " | (best)" * best
        self.logger.info(log_str)
        self.logger.info(CC.c(f"[{epoch}/{self.opt.epochs}] Best mIoU until now: {self.best_iou * 100:.2f}\n", CC.OKGREEN))

        # Log results to the sacred database
        for k, v in accu.mean(self.loss_names, dic=True).items():
            self.run.log_scalar(k, float(v), epoch)
        self.run.log_scalar('val_loss', float(val_loss), epoch)
        self.run.log_scalar('val_mIoU', float(val_mIoU), epoch)
        self.run.log_scalar('val_bIoU', float(val_bIoU), epoch)
        for k, v in kwargs.items():
            self.run.log_scalar(k, float(v), epoch)
    