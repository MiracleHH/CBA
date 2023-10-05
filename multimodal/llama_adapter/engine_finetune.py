import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from data.dataset_vqa import data_post_transform, add_trigger

from llama import LLaMA_adapter

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask, imgs, flags) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
             c_loss, m_loss = model(examples, labels, imgs)
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_backdoor_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, trigger, lower_limit= 0, upper_limit = 1,
                    log_writer=None,
                    args=None, update_trigger = False):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask, imgs, flags) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler

        if (data_iter_step % accum_iter == 0) and not update_trigger:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = imgs.to(device, non_blocking=True)

        if args.attack_type == 'image' or args.attack_type == 'both':
            num_poison = flags.sum()
            num_data = len(flags)
            if num_poison > 0 and num_poison < num_data:
                clean_imgs, clean_labels = imgs[flags == 0], labels[flags == 0]
                backdoor_imgs, backdoor_labels = imgs[flags == 1], labels[flags == 1]
                backdoor_imgs = add_trigger(backdoor_imgs, trigger, args.trig_pos)
                imgs = torch.cat((backdoor_imgs, clean_imgs), dim = 0)
                labels = torch.cat((backdoor_labels, clean_labels), dim = 0)
            elif num_poison == num_data:
                imgs = add_trigger(imgs, trigger, args.trig_pos)

        imgs = data_post_transform(imgs)

        with torch.cuda.amp.autocast():
             c_loss, m_loss = model(examples, labels, imgs)
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        if update_trigger:
            loss_scaler(loss, optimizer, parameters = trigger,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        trigger.data = torch.clamp(trigger, min= lower_limit, max= upper_limit)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}