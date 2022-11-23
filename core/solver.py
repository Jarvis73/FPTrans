import types

import torch
import numpy as np

from config import MapConfig


class PolyLR(object):
    def __init__(self, optimizer, max_iter, power=0.9, lr_end=0, last_step=0):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.power = power
        self.last_step = last_step
        self.init_lr = [group['lr'] for group in optimizer.param_groups]
        if isinstance(lr_end, (int, float)):
            self.lr_end = [lr_end for _ in optimizer.param_groups]
        elif isinstance(lr_end, (tuple, list)):
            self.lr_end = lr_end
            assert len(self.lr_end) == len(self.init_lr), f"{len(self.lr_end)} vs {len(self.init_lr)}"
        else:
            raise TypeError

        self.step()

    def step(self, step=None):
        self.last_step = min(self.last_step + 1, self.max_iter)
        if step is None:
            step = self.last_step
        else:
            step = min(step, self.max_iter)
            self.last_step = step

        for i, (init_lr, lr_end) in enumerate(zip(self.init_lr, self.lr_end)):
            lr = (init_lr - lr_end) * (1 - step / self.max_iter) ** self.power + lr_end
            self.optimizer.param_groups[i]['lr'] = lr

    def state_dict(self):
        return {'last_step': self.last_step}

    def load_state_dict(self, state):
        self.last_step = state['last_step']


class CosineLR(object):
    def __init__(self, optimizer, max_iter, lr_end=0, last_step=0, lr_rev=False):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.last_step = last_step
        self.rev = float(lr_rev)
        self.init_lr = [group['lr'] for group in optimizer.param_groups]
        if isinstance(lr_end, (int, float)):
            self.lr_end = [lr_end for _ in optimizer.param_groups]
        elif isinstance(lr_end, (tuple, list)):
            self.lr_end = lr_end
            assert len(self.lr_end) == len(self.init_lr), f"{len(self.lr_end)} vs {len(self.init_lr)}"
        else:
            raise TypeError

        self.step()

    def step(self, step=None):
        self.last_step = self.last_step + 1
        if step is None:
            step = self.last_step
        else:
            self.last_step = step

        for i, (init_lr, lr_end) in enumerate(zip(self.init_lr, self.lr_end)):
            lr = lr_end + 0.5 * (init_lr - lr_end) * (1 + np.cos(((step - 1) % self.max_iter) / (self.max_iter - 1) * np.pi + np.pi * self.rev))
            self.optimizer.param_groups[i]['lr'] = lr

    def state_dict(self):
        return {'last_step': self.last_step}

    def load_state_dict(self, state):
        self.last_step = state['last_step']


def get(opt, model, max_steps):
    if isinstance(model, list):
        params_group = model
    elif isinstance(model, torch.nn.Module):
        params_group = model.parameters()
    elif isinstance(model, types.GeneratorType):
        params_group = model
    else:
        raise TypeError(f"`model` must be an nn.Model or a list, got {type(model)}")

    if opt.optim == "sgd":
        optimizer_params = {"momentum": opt.sgd_momentum,
                            "weight_decay": opt.weight_decay,
                            "nesterov": opt.sgd_nesterov}
        optimizer = torch.optim.SGD(params_group, opt.lr, **optimizer_params)
    elif opt.optim == "adam":
        optimizer_params = {"betas": (opt.adam_beta1, opt.adam_beta2),
                            "eps": opt.adam_epsilon,
                            "weight_decay": opt.weight_decay}
        optimizer = torch.optim.Adam(params_group, opt.lr, **optimizer_params)
    elif opt.optim == "adamw":
        optimizer_params = {"weight_decay": opt.weight_decay}
        optimizer = torch.optim.AdamW(params_group, opt.lr, **optimizer_params)
    else:
        raise ValueError(f"Not supported optimizer: {opt.optim}")

    if opt.lrp == "period_step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=opt.lr_step,
                                                    gamma=opt.lr_rate)
    elif opt.lrp == "custom_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=opt.lr_boundaries,
                                                         gamma=opt.lr_rate)
    elif opt.lrp == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=opt.lr_rate,
                                                               patience=opt.lr_patience,
                                                               threshold=opt.lr_min_delta,
                                                               cooldown=opt.cool_down,
                                                               min_lr=opt.lr_end)
    elif opt.lrp == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=max_steps,
                                                               eta_min=opt.lr_end)
    elif opt.lrp == "poly":
        scheduler = PolyLR(optimizer,
                           max_iter=max_steps,
                           power=opt.power,
                           lr_end=opt.lr_end)
    elif opt.lrp == 'cosinev2':
        scheduler = CosineLR(optimizer,
                             max_iter=max_steps // opt.lr_repeat,
                             lr_end=opt.lr_end,
                             lr_rev=opt.lr_rev)
    else:
        raise ValueError

    return optimizer, scheduler
