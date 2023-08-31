import pandas as pd
import numpy as np
from os import path
import os
import torch
import time
import datetime
import random
import pytz
from easydict import EasyDict as edict
import yaml
import torch


def save_yaml(config):
    save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)


def set_seed(seed=None):
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed == 'None':
        seed = random.randint(1, 10000)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    return seed


def load_model(exp_dir):
    if os.path.exists(exp_dir):
        ck = torch.load(exp_dir)
        return ck
    else:
        return None


def model_snapshot(epoch, model, optimizer, scheduler, best_valid_loss, exp_dir):
    if scheduler is not None:
        ck = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_valid_loss': best_valid_loss
        }
        torch.save(ck, exp_dir)
    else:
        ck = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_valid_loss': best_valid_loss
        }
        torch.save(ck, exp_dir)


def save_config(config):
    save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
