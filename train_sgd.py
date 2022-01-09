import argparse
import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import utils

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

# Parse arguments
parser = argparse.ArgumentParser(description='Regular training and sampling for DLDR')
parser.add_argument('--arch', '-a', metavar='ARCH',
                    help='The architecture of the model')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('--optimizer',  metavar='OPTIMIZER', default='sgd', type=str,
                    help='The optimizer for training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
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
            

def main():
    # Record training statistics
    best_prec1 = 0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    args = parser.parse_args()
    utils.set_random_seed(args.randomseed)


    # Check the save_dir exists or not
    exp_name = utils.get_exp_name(args, prefix='sgd')
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
    model = torch.nn.DataParallel(utils.get_model(args))
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info("Model = %s" % str(model))
    logging.info('number of params: {} M'.format(n_parameters / 1e6))

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            logging.info(f'from {args.start_epoch}')
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            logging.info("no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Prepare Dataloader
    train_loader, val_loader = utils.get_datasets(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    
    ##################################################################################################
    
    if args.datasets == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150], last_epoch=args.start_epoch - 1)
                                                            
    elif args.datasets == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    is_best = 0
    utils.save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(output_dir, 'checkpoint_refine_' + str(0) + '.th'))
    # DLDR sampling
    torch.save(model.state_dict(), os.path.join(output_dir,  str(0) +  '.pt'))

    logging.info(f'Start training: {args.start_epoch} -> {args.epochs}')
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_stats, train_epoch_loss, train_epoch_acc = train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        val_stats, prec1, test_epoch_loss, test_epoch_acc = validate(args, val_loader, model, criterion, epoch)

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)

        # log metrics to wandb
        log_stats = {'epoch': epoch, 'n_parameters': n_parameters}
        log_stats = dict(train_stats.items() | val_stats.items() | log_stats.items())
        if has_wandb and args.log_wandb:
            wandb.log(log_stats)

        # remember best prec@1 and save checkpoint
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


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    # switch to train mode
    model.train()    
    
    total_loss, total_err = 0, 0
    for i, (input, target) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):

        # Load batch data to cuda
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item() * input_var.shape[0]
        total_err += (output.max(dim=1)[1] != target_var).sum().item()

        optimizer.step()
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
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

def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """

    total_loss = 0
    total_err = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(metric_logger.log_every(val_loader, args.print_freq, header)):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()
                
            total_loss += loss.item() * input_var.shape[0]
            total_err += (output.max(dim=1)[1] != target_var).sum().item()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            metric_logger.update(test_loss=loss.item())
            metric_logger.update(test_prec1=prec1.item())
    
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
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
