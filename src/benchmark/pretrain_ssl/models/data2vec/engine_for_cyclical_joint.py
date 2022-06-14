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
import torch.nn as nn
import torch.nn.functional as F

import utils


def train_one_epoch(model: torch.nn.Module, model_ema: torch.nn.Module, ema_start_at, target_layers,
                    d_vae: torch.nn.Module, vae_loss_weight: float,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, l1_beta: float = 0.12,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, l2_loss=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cyc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_beit', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch

        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        with torch.no_grad():
            targets = model_ema.module(samples, bool_masked_pos=None, return_all_tokens=True, layer_results=True)
            fsz = targets[0].size(-1)

            targets = sum(F.layer_norm(targets[i], (fsz,)) for i in target_layers) / len(target_layers)

            fsz = targets.size(-1)
            target_mask = bool_masked_pos.flatten().bool()
            targets = targets.reshape(-1, fsz)[target_mask]

            # beit part
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]

        with torch.cuda.amp.autocast():
            outputs, beit_outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False)
            outputs = outputs.reshape(-1, fsz)
            assert outputs.shape == targets.shape
            if l2_loss:
                cyc_loss = F.mse_loss(outputs, targets)
            else:
                cyc_loss = F.smooth_l1_loss(outputs, targets, beta=l1_beta)

            # beit part
            beit_loss = nn.CrossEntropyLoss()(input=beit_outputs, target=labels)

        # loss = cyc_loss / (vae_loss_weight + 1) + beit_loss * vae_loss_weight / (vae_loss_weight + 1)
        beit_w = max(1 - (epoch / vae_loss_weight), 0)
        loss = cyc_loss * (1 - beit_w) + beit_loss * beit_w
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if it == ema_start_at and ema_start_at > 0:
            print(f"setting EMA to model params at update {it}")
            model_ema.set(model)
        elif it >= ema_start_at:
            model_ema.update(model)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(loss_cyc=cyc_loss.item())
        metric_logger.update(loss_beit=beit_loss.item())
        # metric_logger.update(loss_cyc=cyc_loss.item(), head="loss_cyc")
        # metric_logger.update(loss_beit=beit_loss.item(), head="loss_beit")

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss=cyc_loss.item(), head="loss_cyc")
            log_writer.update(loss=beit_loss.item(), head="loss_beit")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
