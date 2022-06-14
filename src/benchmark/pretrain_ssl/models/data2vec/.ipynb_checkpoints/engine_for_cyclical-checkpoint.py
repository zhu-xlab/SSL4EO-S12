# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

from . import utils


def train_one_epoch(model: torch.nn.Module, model_ema: torch.nn.Module, ema_start_at, decay_init, decay, target_layers,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,     
                    l1_beta: float = 0.12,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, l2_loss=False, layer_results='end',
                    var_w0=0, var_w1=0, var_margin0=0.5, var_margin1=0.5, start_lr_decay_at_step=-1,loss_scale=-1, mask_dropout_prob=-1.0,
                    target_layer_norm_last=True, target_batch_norm=False, target_instance_norm=False,post_target_instance_norm=False,post_target_layer_norm=False):
    print(' <<<<<<<< layer_results >>>>>>>>', layer_results)
    print(' <<<<<<<< var_w0, var_w1 >>>>>>>>', var_w0, var_w1)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_var0', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('loss_var1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    cur_decay = decay
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    #for batch in data_loader:
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if it < ema_start_at:
            cur_decay = decay_init + it * (decay - decay_init) / ema_start_at 
        
        samples, bool_masked_pos = batch
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        if mask_dropout_prob > 0:
            new_mask_tensor = torch.ones_like(bool_masked_pos, dtype=samples.dtype)
            new_mask_tensor.fill_(1-mask_dropout_prob)
            bool_new_mask_tensor = torch.bernoulli(new_mask_tensor)
            bool_masked_pos = torch.logical_and(bool_new_mask_tensor, bool_masked_pos)
        
        with torch.no_grad():
            targets = model_ema.module(samples, bool_masked_pos=None, return_all_tokens=True, layer_results=layer_results)
            fsz = targets[0].size(-1)
            #shape of targets[0] == b x t x dim
            layer_vals = [targets[i] for i in target_layers]
            
            if target_instance_norm or target_batch_norm:
                layer_vals = [val.permute(0,2,1) for val in layer_vals] # btc => bct

            if target_batch_norm:
                layer_vals = [F.batch_norm(val.float(), running_mean=None, running_var=None, training=True) for val in layer_vals] # bct => bct

            if target_instance_norm:
                layer_vals = [F.instance_norm(val.float()) for val in layer_vals] # bct => bct
            
            if target_instance_norm or target_batch_norm:
                layer_vals = [val.permute(0,2,1) for val in layer_vals] # bct => btc

            if target_layer_norm_last:
                layer_vals = (F.layer_norm(val.float(), (fsz,)) for val in layer_vals)

            targets = sum(layer_vals) / len(target_layers)

            if post_target_instance_norm:
                targets = targets.permute(0,2,1)
                targets = F.instance_norm(targets.float())
                targets = targets.permute(0,2,1)

            if post_target_layer_norm:
                targets = F.layer_norm(targets.float(), (fsz,))

            fsz = targets.size(-1)
            target_mask = bool_masked_pos.flatten().bool()
            targets = targets.reshape(-1, fsz)[target_mask]

        with torch.cuda.amp.autocast():
            outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False)

        outputs = outputs.float()

        eps=1e-6
        z0 = outputs.reshape(-1, outputs.size(-1))
        z0 = torch.sqrt(z0.var(dim=0) + eps)

        if var_w0 > 0:
            std_loss0 = torch.sum(F.relu(var_margin0 - z0)) / z0.size(0)
        else:
            std_loss0 = 0

        # z1 = torch.sqrt(outputs.var(dim=1) + eps)
        # std_loss1 = torch.sum(F.relu(var_margin1 - z1)) / outputs.size(0)

        # print(outputs.shape)
        outputs = outputs.reshape(-1, fsz)
        assert outputs.shape == targets.shape
        if l2_loss:
            loss_cyc = F.mse_loss(outputs, targets)
        else:
            loss_cyc = F.smooth_l1_loss(outputs, targets, beta=l1_beta)

        # loss = loss_cyc + std_loss0 * var_w0 + std_loss1 * var_w1
        loss = loss_cyc + std_loss0 * var_w0 
        if loss_scale!=-1:
            loss = loss * loss_scale
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # if it == ema_start_at and ema_start_at > 0:
        #     print(f"setting EMA to model params at update {it}")
        #     model_ema.set(model)
        # elif it >= ema_start_at:
        #     model_ema.update(model)
        if cur_decay!=1 and (start_lr_decay_at_step==-1 or it<=start_lr_decay_at_step):
            model_ema._update(model, update_fn=lambda e, m: cur_decay * e + (1. - cur_decay) * m)
        else:
            cur_decay=0
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(loss_var0=std_loss0)
        # metric_logger.update(loss_var1=std_loss1)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(cur_decay=cur_decay)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            # log_writer.update(std_loss0=std_loss0.item(), head="std_loss0")
            # log_writer.update(std_loss1=std_loss1.item(), head="std_loss1")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(cur_decay=cur_decay, head="cur_decay")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
