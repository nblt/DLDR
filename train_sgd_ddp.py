import argparse
import datetime
import os
import time
import math
import sys
import logging
import numpy as np
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
import model_dldr

import timm.scheduler, timm.optim

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

# Parse arguments
parser = argparse.ArgumentParser(description='Regular training and sampling for DLDR')
# arch
parser.add_argument('--arch', '-a', metavar='ARCH',
                    help='The architecture of the model')
# dataset
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                    help='')
parser.set_defaults(pin_mem=True)
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    weight decay. We use a cosine schedule for WD. 
    (Set the same value with args.weight_decay to keep weight decay no change)""")

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr_cycle_mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr_cycle_decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr_cycle_limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr_k_decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup_lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epoch_repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--decay_epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# training
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50 iterations)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save_dir', default='save_temp', type=str,
                    help='The directory used to save the trained models')
parser.add_argument('--save_every', type=int, default=10,
                    help='Saves checkpoints at every specified number of epochs')
parser.add_argument('--step_sample_freq', type=int, default=50,
                    help='frequency to sample model params in a mini-batch')
parser.add_argument('--randomseed', type=int, default=1, 
                    help='Randomseed for training and initialization')
parser.add_argument('--corrupt', default=0, type=float,
                    metavar='c', help='noise level for training set')
parser.add_argument('--smalldatasets', default=None, type=float, dest='smalldatasets', 
                    help='percent of small datasets')
# wandb log
parser.add_argument('--project', default='', type=str, metavar='NAME',
                help='name of wandb project')
parser.add_argument('--log_wandb', action='store_true', default=False,
                help='log training and validation metrics to wandb')
# distributed training parameters
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
            

def main():
    # Record training statistics
    best_prec1 = 0
    is_best = 0
    train_acc, test_acc, train_loss, test_loss = [], [], [], []
    
    args = parser.parse_args()
    utils.set_random_seed(args.randomseed)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    # Prepare Data Loader
    dataset_train, dataset_test = utils.get_datasets_ddp(args)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train, data_loader_test = utils.get_loader_ddp(args, dataset_train, dataset_test, sampler_train)

    # Define model
    model = utils.get_model(args)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * max(1.0, total_batch_size / 256)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    if utils.is_main_process():
        # Check the save_dir exists or not
        exp_name = utils.get_exp_name(args, prefix='sgd_ddp')
        output_dir = utils.get_outdir(args.save_dir if args.save_dir else './output', exp_name)
        print(f"save at {output_dir}")
        utils.dump_args(args, output_dir)
        logFilename = os.path.join(output_dir, "train.log")
        print(f"logging into: {logFilename}")
        utils.console_out(logFilename)

        # Initialize wandb to log metrics
        if args.log_wandb:
            if has_wandb:
                wandb.init(project=args.project, config=args)
                wandb.run.name = exp_name
            else: 
                print("You've requested to log metrics to wandb but package not found. "
                                "Metrics not being logged to wandb, try `pip install wandb`")
        # DLDR sampling
        utils.sample_model(epoch=0, output_dir=output_dir, model_without_ddp=model_without_ddp)

    # # Optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logging.info("loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         logging.info(f'from {args.start_epoch}')
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         logging.info("loaded checkpoint '{}' (epoch {})"
    #               .format(args.evaluate, checkpoint['epoch']))
    #     else:
    #         logging.info("no checkpoint found at '{}'".format(args.resume))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    # if args.opt == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)
    # elif args.opt == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    optimizer = timm.optim.create_optimizer(args, model_without_ddp)
    
    ##################################################################################################
    
    # if args.datasets == 'CIFAR10':
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                         milestones=[100, 150], last_epoch=args.start_epoch - 1)
                                                            
    # elif args.datasets == 'CIFAR100':
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                         milestones=[150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    # print("Use step level LR & WD scheduler!")
    # lr_schedule_values = utils.cosine_scheduler(
    #     args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
    #     warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    # )
    # if args.weight_decay_end is None:
    #     args.weight_decay_end = args.weight_decay
    # wd_schedule_values = utils.cosine_scheduler(
    #     args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    # print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = timm.scheduler.create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    # loss_scaler = NativeScaler()
    loss_scaler = None
        
    logging.info("Model = %s" % str(model))
    logging.info('number of params: {} M'.format(n_parameters / 1e6))
    logging.info("Batch size = %d" % total_batch_size)
    logging.info("Number of training steps = %d" % num_training_steps_per_epoch)
    logging.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))
    logging.info(f'Start training: {start_epoch} -> {num_epochs}')
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        if args.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        
        if args.sched == "onecycle" and lr_scheduler is not None:
            lr_scheduler.step(epoch)

        # train for one epoch
        train_stats, train_epoch_loss, train_epoch_acc = train(args, 
            data_loader_train, model, criterion, optimizer, epoch, 
            loss_scaler, device, args.clip_grad, 
            start_steps=epoch * num_training_steps_per_epoch, 
            output_dir=output_dir)

        # evaluate on validation set
        val_stats, prec1, test_epoch_loss, test_epoch_acc = validate(args, 
            data_loader_test, model, criterion, epoch, device)

        if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, val_stats["test_loss"])

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if utils.is_main_process():
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            test_loss.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)
            # log metrics to wandb
            log_stats = {'epoch': epoch, 'n_parameters': n_parameters}
            log_stats = dict(train_stats.items() | val_stats.items() | log_stats.items())
            if has_wandb and args.log_wandb:
                wandb.log(log_stats)
            logging.info(f"\033[0;36m @best prec1: {best_prec1} \033[0m")

        if utils.is_main_process() and (epoch > 0 and epoch % args.save_every == 0 or epoch == args.epochs - 1):
            utils.save_model(
                args=args, epoch=epoch, output_dir=output_dir, 
                model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler
            )
        if utils.is_main_process():
            # DLDR sampling
            utils.sample_model(epoch=epoch+1, output_dir=output_dir, model_without_ddp=model_without_ddp)

    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f'total time: {total_time_str}')
        logging.info(f'best_prec1: {best_prec1}')
        utils.log_dump_metrics(output_dir=output_dir,
            train_loss=train_loss, train_acc=train_acc, 
            test_loss=test_loss, test_acc=test_acc
        ) 


def train(args, train_loader: Iterable, model: torch.nn.Module, criterion, 
          optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, 
          device: torch.device, max_norm: float=0, lr_scheduler=None, 
          start_steps=None, lr_schedule_values=None, wd_schedule_values=None, 
          output_dir=None):
    """
    Run one train epoch
    """
    # switch to train mode
    model.train() 
    model_without_ddp = model.module

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)   
    num_updates = epoch * len(train_loader)
    for step, (input, target) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        if lr_scheduler is not None:
            lr_scheduler.step_frac(epoch + (step + 1) / len(train_loader))
        # Load batch data to cuda
        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        with torch.cuda.amp.autocast():
            output = model(input_var)
            loss = criterion(output, target_var)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = None
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, 
                                    clip_grad=max_norm,
                                    parameters=model.parameters(), 
                                    create_graph=is_second_order)
        else:
            loss.backward(create_graph=is_second_order)
            optimizer.step()

        # optimizer.step()
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        torch.cuda.synchronize()
        num_updates += 1
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(train_max_lr=max_lr)
        metric_logger.update(train_min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]

        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_prec1=prec1.item())
        # metric_logger.update(train_lr=optimizer.param_groups[0]['lr'])
        if weight_decay_value is not None:
            metric_logger.update(weight_decay=weight_decay_value)
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

    # lr_scheduler.step()
        # if lr_scheduler is not None:
        #     lr_scheduler.step_update(start_steps + step)
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=metric_logger.meters['train_loss'].global_avg)

        if output_dir is not None and (
            step > 0 and step % args.step_sample_freq == 0 or step == len(train_loader) - 1
            ) and utils.is_main_process():
            # DLDR sampling
            utils.sample_model(epoch=epoch+1, output_dir=output_dir, model_without_ddp=model_without_ddp, step=step+1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged train stats: {metric_logger}")

    train_epoch_loss = metric_logger.meters['train_loss'].global_avg
    train_epoch_acc = metric_logger.meters['train_prec1'].global_avg

    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 
        train_epoch_loss, 
        train_epoch_acc
    )

def validate(args, val_loader: Iterable, model: torch.nn.Module, criterion, 
             epoch: int, device: torch.device):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    with torch.no_grad():
        for step, (input, target) in enumerate(metric_logger.log_every(val_loader, args.print_freq, header)):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target

            if args.half:
                input_var = input_var.half()

            # compute output
            with torch.cuda.amp.autocast():
                output = model(input_var)
                loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            torch.cuda.synchronize()

            metric_logger.update(test_loss=loss.item())
            metric_logger.update(test_prec1=prec1.item())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged train stats: {metric_logger}")

    test_epoch_loss = metric_logger.meters['test_loss'].global_avg
    test_epoch_acc = metric_logger.meters['test_prec1'].global_avg

    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 
        metric_logger.meters['test_prec1'].global_avg, 
        test_epoch_loss, 
        test_epoch_acc
    )


if __name__ == '__main__':
    main()
