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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pickle
import random
from model_dldr import reparam_model_v2
import resnet

import utils

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
parser.add_argument('--epochs', default=100, type=int, metavar='N',
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
parser.add_argument('--n_components', default=40, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which epoch start for PCA') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='which epoch end for PCA') 
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='lr for PSGD') 
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
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
    exp_name = utils.get_exp_name(args, prefix='psgd_ddp_v2')
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
    model = utils.get_model(args)
    model = torch.nn.DataParallel(model)
    # model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load sampled model parameters
    logging.info(f'params: from {args.params_start} to {args.params_end}')
    # group1: other than linear , group2: linear
    group_W = [[], []]
    for i in range(args.params_start, args.params_end):
        ############################################################################
        # if i % 2 != 0: continue

        model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, str(i) + '.pt')))
        vec = [[], []]
        for name, param in model.named_parameters():
            if "linear" in name or "fc" in name:
                vec[1].append(param.detach().cpu().numpy().reshape(-1))
            else:
                vec[0].append(param.detach().cpu().numpy().reshape(-1))

        for i, v in enumerate(vec):
            group_W[i].append(np.concatenate(v, 0))
    
    group_P = []
    group_n_components = []
    for i, W in enumerate(group_W):
        logging.info(f"group {i}")
        W = np.array(W)
        logging.info(f'W: {W.shape}')

        group_n_components.append(args.n_components)
        # Obtain base variables through PCA
        pca = PCA(n_components=args.n_components)
        pca.fit_transform(W)
        P = np.array(pca.components_)
        # np.save(os.path.join(output_dir, f"P_{args.params_start}_{args.params_end}_{args.n_components}.npy"), P)
        logging.info(f'ratio: {pca.explained_variance_ratio_}')
        logging.info(f'P: {P.shape}')
        P = torch.from_numpy(P).cuda()
        group_P.append(P)

    # Resume from params_start
    model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, str(0) + '.pt')))

    group_param0 = [[], []]
    for name, param in model.named_parameters():
        if "linear" in name or "fc" in name:
            group_param0[1].append(param.detach().cpu().numpy().reshape(-1))
        else:
            group_param0[0].append(param.detach().cpu().numpy().reshape(-1))
    group_param0[0] = torch.from_numpy(np.concatenate(group_param0[0], 0)).cuda()
    group_param0[1] = torch.from_numpy(np.concatenate(group_param0[1], 0)).cuda()

    del model

    # Build reparameterize model
    model = utils.get_model(args)
    group_params = [[], []]
    other_params = []
    for name, p in model.named_parameters():
        if "linear" in name or "fc" in name:
            group_params[1].append(p)
        else:
            group_params[0].append(p)

    reparam_model = reparam_model_v2(model=model, param0=group_param0, n_components=group_n_components, 
                                     group_params=group_params, other_params=other_params, P=group_P)

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

    if len(other_params):
        optimizer = optim.SGD([
            {'params': reparam_model.module.get_param()},
            {'params': other_params, 'lr': args.lr, 'momentum': args.momentum}], 
            lr=args.lr, 
            momentum=args.momentum)
    else:
        optimizer = optim.SGD(reparam_model.module.get_param(), lr=args.lr, momentum=args.momentum)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], last_epoch=args.start_epoch - 1)

    logging.info(f'Start training: {args.start_epoch} -> {args.epochs}')
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_stats, train_epoch_loss, train_epoch_acc = train(args, train_loader, reparam_model, criterion, optimizer, epoch)
        # Bk = torch.eye(args.n_components).cuda()
        lr_scheduler.step()

        # evaluate on validation set
        val_stats, prec1, test_epoch_loss, test_epoch_acc = validate(args, val_loader, reparam_model, criterion, epoch)

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

def train(args, train_loader, model, criterion, optimizer, epoch):
    # Run one train epoch
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    # Switch to train mode
    model.train()

    for i, (input, target) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):

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
        # Measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_prec1=prec1.item())
        metric_logger.update(train_lr=optimizer.param_groups[0]['lr'])

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
        for i, (input, target) in enumerate(metric_logger.log_every(val_loader, args.print_freq, header)):
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