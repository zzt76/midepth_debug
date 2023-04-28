import os
from sched import scheduler

import torch


def save_weights(model, filename, folder="./saved_models"):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    fpath = os.path.join(folder, filename)
    torch.save({"model": model.state_dict()}, fpath)
    return


def save_checkpoint(model, optimizer, scheduler, epoch, filename, folder="./checkpoints", save_weights_only=False):
    if save_weights_only:
        save_weights(model, filename, folder)

    else:
        if not os.path.isdir(folder):
            os.makedirs(folder)

        fpath = os.path.join(folder, filename)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }, fpath)


def load_weights(model, filename, path="./saved_models"):
    fpath = os.path.join(path, filename)
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model


def load_weights_from_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')
    ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
    model.load_state_dict(load_dict)
    return model


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = None
    if 'scheduler' in ckpt:
        scheduler = ckpt['scheduler']

    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model, optimizer, scheduler, epoch


def load_checkpoint_finetune(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    scheduler = None
    epoch = 0

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model, optimizer, scheduler, epoch
