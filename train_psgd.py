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
parser.add_argument('--alpha', default=0, type=float, metavar='N',
                    help='lr for momentum') 
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='lr for PSGD') 
parser.add_argument('--gamma', default=0.9, type=float, metavar='N',
                    help='gamma for momentum')
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

args = parser.parse_args()
utils.set_random_seed(args.randomseed)
best_prec1 = 0
P = None
train_acc, test_acc, train_loss, test_loss = [], [], [], []

def main():

    global args, best_prec1, Bk, p0, P

    # Check the save_dir exists or not
    exp_name = utils.get_exp_name(args, prefix='psgd')
    output_dir = utils.get_outdir(args.save_dir if args.save_dir else './output', exp_name)
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
    model = torch.nn.DataParallel(utils.get_model(args))
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info("Model = %s" % str(model))
    logging.info('number of params: {} M'.format(n_parameters / 1e6))

    # Load sampled model parameters
    logging.info(f'params: from {args.params_start} to {args.params_end}')
    W = []
    for i in range(args.params_start, args.params_end):
        ############################################################################
        # if i % 2 != 0: continue

        model.load_state_dict(torch.load(os.path.join(args.pretrain_dir,  str(i) +  '.pt')))
        W.append(utils.get_model_param_vec(model))
    W = np.array(W)
    logging.info(f'W: {W.shape}')

    # Obtain base variables through PCA
    pca = PCA(n_components=args.n_components)
    pca.fit_transform(W)
    P = np.array(pca.components_)
    np.save(os.path.join(output_dir, f"P_{args.params_start}_{args.params_end}_{args.n_components}.npy"), P)
    logging.info(f'ratio: {pca.explained_variance_ratio_}')
    logging.info(f'P: {P.shape}')

    P = torch.from_numpy(P).cuda()

    # Resume from params_start
    model.load_state_dict(torch.load(os.path.join(args.pretrain_dir,  str(args.params_start) +  '.pt')))

    # Prepare Dataloader
    train_loader, val_loader = utils.get_datasets(args)
    
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()

    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 50], last_epoch=args.start_epoch - 1)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    logging.info(f'Train: {args.start_epoch} + {args.epochs}')
    end = time.time()
    p0 = utils.get_model_param_vec(model)
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch)
        # Bk = torch.eye(args.n_components).cuda()
        lr_scheduler.step()

        # evaluate on validation set
        prec1, val_stats = validate(val_loader, model, criterion, epoch)

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
    logging.info(f'train loss: {train_loss}')
    logging.info(f'train acc: {train_acc}')
    logging.info(f'test loss: {test_loss}')
    logging.info(f'test acc: {test_acc}')      
    logging.info(f'best_prec1: {best_prec1}')

    torch.save(model.state_dict(), os.path.join(output_dir, 'PSGD.pt'))

running_grad = 0

def train(train_loader, model, criterion, optimizer, epoch):
    # Run one train epoch

    global P, W, iters, T, train_loss, train_acc, search_times, running_grad, p0
    
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
        optimizer.zero_grad()
        loss.backward()

        # Do P_plus_BFGS update
        grad = utils.get_model_grad_vec(model)
        P_SGD(model, optimizer, grad)

        # Measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_prec1=prec1.item())
        metric_logger.update(train_lr=optimizer.param_groups[0]['lr'])
        
        # if i % args.print_freq == 0 or i == len(train_loader)-1:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #               epoch, i, len(train_loader), batch_time=batch_time,
        #               data_time=data_time, loss=losses, top1=top1))

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    logging.info(f"Averaged train stats: {metric_logger}")    
    
    train_loss.append(metric_logger.meters['train_loss'].global_avg)
    train_acc.append(metric_logger.meters['train_prec1'].global_avg)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def P_SGD(model, optimizer, grad):
    # P_SGD algorithm

    gk = torch.mm(P, grad.reshape(-1,1))

    grad_proj = torch.mm(P.transpose(0, 1), gk)
    grad_res = grad - grad_proj.reshape(-1)

    # Update the model grad and do a step
    utils.update_grad(model, grad_proj)
    optimizer.step()

def validate(val_loader, model, criterion, epoch):
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
            metric_logger.update(val_loss=loss.item())
            metric_logger.update(val_prec1=prec1.item())

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1))

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    logging.info(f"Averaged train stats: {metric_logger}")

    # Store the test loss and test accuracy
    test_loss.append(metric_logger.meters['val_loss'].global_avg)
    test_acc.append(metric_logger.meters['val_prec1'].global_avg)

    return metric_logger.meters['val_prec1'].global_avg, {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
