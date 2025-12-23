import numpy as np
import torch
import torch.optim
from transformers import get_scheduler
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers.utils import logging


def get_optimizer(model, config):
    attn_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "attn" in name:
            attn_params.append(param)
            logging.info(f"Attn params: {name}, lr: {config.train.lr_attn}")
        else:
            other_params.append(param)
            logging.info(f"Other params: {name}, lr: {config.train.lr}")
    optimizer_grouped_parameters = [
        {"params": attn_params, "lr": config.train.lr_attn},
        {"params": other_params, "lr": config.train.lr},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), fused=True)
    return optimizer


def count_model_params(model, requires_grad: bool = True):
    # code form lolcats
    """
    Return total # of trainable parameters
    """
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    try:
        return sum([np.prod(p.size()) for p in model_parameters]).item()
    except:
        return sum([np.prod(p.size()) for p in model_parameters])

