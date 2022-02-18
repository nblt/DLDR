import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pickle
import random
from model_dldr import reparam_model_v1
import resnet

import utils
import timm.scheduler, timm.optim

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

parser = argparse.ArgumentParser(description='P(+)-SGD in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    help='model architecture (default: resnet32)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save_dir', default='save_temp', type=str ,
                    help='The directory used to save training')
parser.add_argument('--pretrain_dir', default='save_temp', type=str,
                    help='The directory used to save the pretrained models')
parser.add_argument('--save_every', type=int, default=10 ,
                    help='Saves checkpoints at every specified number of epochs')
# DLDR sample setting 
parser.add_argument('--sample_mode', default="", type=str ,
                    help='the mode to sample parameters for pca')
parser.add_argument('--sample_beta', type=float, default=0.8, metavar='M',
                    help='beta for exp smooth sampling (default: 0.8)')
parser.add_argument('--save_pca_p', dest='save_pca_p', action='store_true',
                    help='save the pca transform matrix')
parser.add_argument('--n_components', default=40, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which epoch start for PCA') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='which epoch end for PCA') 
parser.add_argument('--dldr_start', default=-1, type=int, metavar='N',
                    help='which epoch start dldr train') 

# Optimizer parameters
parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='Optimizer learning rate') 
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
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
parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

parser.add_argument('--randomseed', 
                    help='Randomseed for training and initialization',
                    type=int, default=1)

parser.add_argument('--corrupt', default=0, type=float,
                    metavar='c', help='noise level for training set')
parser.add_argument('--smalldatasets', default=None, type=float, dest='smalldatasets', 
                    help='percent of small datasets')

# wandb log
parser.add_argument('--project', default='', type=str, metavar='NAME',
                help='name of wandb project')
parser.add_argument('--log_wandb', action='store_true', default=False,
                help='log training and validation metrics to wandb')


def main():
    P = None
    # Record training statistics
    best_prec1 = 0
    train_acc, test_acc, train_loss, test_loss = [], [], [], []

    args = parser.parse_args()
    utils.set_random_seed(args.randomseed)

    # Check the save_dir exists or not
    exp_name = utils.get_exp_name(args, prefix='psgd_ddp')
    output_dir = utils.get_outdir(args.save_dir if args.save_dir else './output', exp_name)
    print(f"save at {output_dir}")
    utils.dump_args(args, output_dir)
    logFilename = os.path.join(output_dir, "train.log")
    utils.console_out(logFilename)

    # Initialize wandb to log metrics
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.project, config=args)
            wandb.run.name = exp_name
        else: 
            print("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
    
    # Define model
    try:
        model = utils.get_model(args)
        # model = torch.nn.DataParallel(model)
        model.cuda()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Load sampled model parameters
        W = utils.get_W(args, model, mode=args.sample_mode, beta=args.sample_beta)
    except:
        model = utils.get_model(args)
        logging.info(f'use DataParallel to wrap model for sampling')
        model = torch.nn.DataParallel(model)
        model.cuda()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Load sampled model parameters
        W = utils.get_W(args, model, mode=args.sample_mode, beta=args.sample_beta)

    # Obtain base variables through PCA
    if args.save_pca_p:
        P, pca = utils.get_P(args, W, output_dir)
    else:
        P, pca = utils.get_P(args, W)
    P = torch.from_numpy(P)# .cuda()

    # Resume from params_start or selected dldr_start
    if args.dldr_start < 0:
        model.load_state_dict(torch.load(os.path.join(args.pretrain_dir,  str(args.params_start) +  '.pt')))
    else:
        model.load_state_dict(torch.load(os.path.join(args.pretrain_dir,  str(args.dldr_start) +  '.pt')))
    param0 = torch.from_numpy(utils.get_model_param_vec(model))# .cuda()

    del model

    # Build reparameterize model
    model = utils.get_model(args)
    reparam_model = reparam_model_v1(model=model, param0=param0, n_components=args.n_components, P=P)
    reparam_model = torch.nn.DataParallel(reparam_model)
    reparam_model.cuda()
    logging.info("Reparam Model = %s" % str(reparam_model))
    logging.info('number of params: {} M'.format(n_parameters / 1e6))
    logging.info(f'n components = {args.n_components}')

    # Prepare Dataloader
    train_loader, val_loader = utils.get_datasets(args)
    
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()

    cudnn.benchmark = True

    if args.opt == "momentum":
        optimizer = optim.SGD(reparam_model.module.get_param(), lr=args.lr, momentum=args.momentum)
    elif args.opt == "adam":
        optimizer = optim.Adam(reparam_model.module.get_param(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(reparam_model.module.get_param(), lr=args.lr)
    
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], last_epoch=args.start_epoch - 1)

    lr_scheduler, num_epochs = timm.scheduler.create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    logging.info('Scheduled epochs: {}'.format(num_epochs))

    logging.info(f'Start training: {start_epoch} -> {num_epochs}')
    end = time.time()
    for epoch in range(start_epoch, num_epochs):

        if args.sched == "onecycle" and lr_scheduler is not None:
            lr_scheduler.step(epoch)

        # train for one epoch
        train_stats, train_epoch_loss, train_epoch_acc = train(args, train_loader, reparam_model, criterion, optimizer, epoch, lr_scheduler)
        # Bk = torch.eye(args.n_components).cuda()

        # evaluate on validation set
        val_stats, prec1, test_epoch_loss, test_epoch_acc = validate(args, val_loader, reparam_model, criterion, epoch)

        if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, val_stats["test_loss"])

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)

        # log metrics to wandb
        log_stats = {'epoch': epoch, 'n_parameters': n_parameters}
        log_stats = dict(train_stats.items() | val_stats.items() | log_stats.items())
        if has_wandb and args.log_wandb:
            wandb.log(log_stats)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        logging.info(f"\033[0;36m @best prec1: {best_prec1} \033[0m")

        if epoch > 0 and epoch % args.save_every == 0 or epoch == args.epochs - 1:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(output_dir, 'checkpoint_refine_' + str(epoch+1) + '.th'))

        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(output_dir, 'model.th'))

        # DLDR sampling
        torch.save(model.state_dict(), os.path.join(output_dir,  str(epoch + 1) +  '.pt'))

    logging.info(f'total time: {time.time() - end}')
    logging.info(f'best_prec1: {best_prec1}')
    utils.log_dump_metrics(output_dir=output_dir,
        train_loss=train_loss, train_acc=train_acc, 
        test_loss=test_loss, test_acc=test_acc
    )   

    torch.save(model.state_dict(), os.path.join(output_dir, 'PSGD.pt'))

def train(args, train_loader, model, criterion, optimizer, epoch, lr_scheduler=None):
    # Run one train epoch
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    # Switch to train mode
    model.train()
    num_updates = epoch * len(train_loader)
    for step, (input, target) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        
        if lr_scheduler is not None:
            lr_scheduler.step_frac(epoch + (step + 1) / len(train_loader))
        # Load batch data to cuda
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Compute gradient and do SGD step
        model.module.model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        model.module.update_low_dim_grad()

        optimizer.step()

        output = output.float()
        loss = loss.float()
        num_updates += 1
        # Measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_prec1=prec1.item())
        metric_logger.update(train_lr=optimizer.param_groups[0]['lr'])

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=metric_logger.meters['train_loss'].global_avg)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    logging.info(f"Averaged train stats: {metric_logger}")    
    
    train_epoch_loss = metric_logger.meters['train_loss'].global_avg
    train_epoch_acc = metric_logger.meters['train_prec1'].global_avg

    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        train_epoch_loss, 
        train_epoch_acc
    )

def P_SGD(model, optimizer, grad, P):
    # P_SGD algorithm

    gk = torch.mm(P, grad.reshape(-1,1))

    grad_proj = torch.mm(P.transpose(0, 1), gk)
    grad_res = grad - grad_proj.reshape(-1)

    # Update the model grad and do a step
    utils.update_grad(model, grad_proj)
    optimizer.step()

def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    global test_acc, test_loss  

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(metric_logger.log_every(val_loader, args.print_freq, header)):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # Compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # Measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            metric_logger.update(test_loss=loss.item())
            metric_logger.update(test_prec1=prec1.item())

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    logging.info(f"Averaged train stats: {metric_logger}")

    # Store the test loss and test accuracy
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
