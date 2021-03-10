import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from numpy import linalg as LA

from sklearn.decomposition import PCA
import numpy as np
import pickle
import random
import models

def set_seed(seed=233): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

parser = argparse.ArgumentParser(description='P+-BFGS in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    help='model architecture (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--n_components', default=10, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='how many params for PCA') 
parser.add_argument('--alpha', default=0, type=float, metavar='N',
                    help='lr for momentum') 

args = parser.parse_args()
best_prec1 = 0
iters = 0
train_acc, test_acc, train_loss, test_loss = [], [], [], []

# Load sampled model parameters
f = open('../save_'+args.arch+'/param_data_100.txt','rb')  
# f = open('./save_nobn_resnet20/param_data_100_100.txt','rb')  
data = pickle.load(f)
print ('params_end', args.params_end)
W = data[:args.params_end, :]
# W = data
print ('W:', W.shape)
f.close()

# Obtain basis variables through PCA
pca = PCA(n_components=args.n_components)
pca.fit_transform(W)
P = np.array(pca.components_)
print ('P:', P.shape)

W = None

def get_model_param_vec(model):
    # Return the model parameters as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().reshape(-1).numpy())
    return np.concatenate(vec, 0)

def get_model_grad_vec(model):
    # Return the model grad as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.grad.detach().cpu().reshape(-1).numpy())
    return np.concatenate(vec, 0)

def update_grad(model, grad_vec):
    # Update the model parameters according to grad_vec

    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.grad = torch.from_numpy(grad_vec[idx:idx+size].reshape(arr_shape)).cuda().float()
        idx += size
        assert param.grad.shape == arr_shape

param_vecs_idx = 0
def add_param_vecs(param_vec):
    # Maintain a queue for storing sampled model parameters

    global W, args, param_vecs_idx, pca, P, Bk, grad_res_momentum

    SIZE = 100
    if W is None:
        W = np.array([param_vec])
    elif len(W) < SIZE:
        W = np.row_stack((W, param_vec))
    else:
        W[param_vecs_idx, :] = param_vec
        param_vecs_idx = (param_vecs_idx + 1) % SIZE
    
def re_compute():
    # Recompute the basis variables

    global W, P, Bk, grad_res_momentum

    pca.fit_transform(W)
    P = np.array(pca.components_)
    print ('P:', P.shape)

    # Flush the inverse Hessian approximation
    Bk = np.eye(args.n_components)
    grad_res_momentum = 0

def main():

    global args, best_prec1, Bk

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Define model
    # model = resnet.resnet20()
    # model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model = torch.nn.DataParallel(models.__dict__[args.arch]())
    model.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            print ('from ', args.start_epoch)

            best_prec1 = checkpoint['best_prec1']
            print ('best_prec:', best_prec1)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Prepare dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='../data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    print ('train batch size!!!', args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True) 
    
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0)
    # The original optimzer
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer.zero_grad()
    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    print ('Train:', (args.start_epoch, args.epochs))
    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # Bk = np.eye(args.n_components)
        fflush()

        # Evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_final_' + str(epoch+1) + '.th'))


        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    print ('train loss: ', train_loss)
    print ('train acc: ', train_acc)
    print ('test loss: ', test_loss)
    print ('test acc: ', test_acc)      

running_grad = 0

def train(train_loader, model, criterion, optimizer, epoch):
    # Run one train epoch

    global P, W, iters, T, train_loss, train_acc, search_times, running_grad
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # Measure data loading time
        data_time.update(time.time() - end)

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
        
        
        iters += 1
        loss.backward()
        if (iters % 8 != 0): continue
        
        # Do P_plus_BFGS update
        gk = get_model_grad_vec(model) / 8.
        P_plus_BFGS(model, optimizer, gk, loss.item(), input_var, target_var)
        # optimizer.step()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

            
        # if iters % T == 0:
        #     re_compute()
        
        optimizer.zero_grad()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
    train_loss.append(losses.avg)
    train_acc.append(top1.avg)

    # print (search_times)
    # search_times = []

# Set the update period of basis variables (per iterations)
T = 1000

# Set the momentum parameters
# gamma = 0.9
gamma = 0.9
# for no bn
# alpha = 0.0001
alpha = args.alpha
print ('alpha:', alpha)
grad_res_momentum = 0

# Store the last gradient on basis variables for P_plus_BFGS update
gk_last = None

# Variables for BFGS and backtracking line search
rho = 0.55
sigma = 0.4
Bk = np.eye(args.n_components)
sk = None

# Store the backtracking line search times
search_times = []


def fflush():
    global Bk, args, sk, gk_last
    Bk = np.eye(args.n_components)
    sk = None
    gk_last = None

def P_plus_BFGS(model, optimizer, grad, oldf, X, y):
    # P_plus_BFGS algorithm

    global rho, sigma, Bk, sk, gk_last, grad_res_momentum, gamma, alpha, search_times

    gk = np.mat(np.matmul(P, grad)).T
    grad_proj = np.matmul(P.transpose(), gk.A.reshape(-1))
    grad_res = grad - grad_proj

    # Quasi-Newton update
    if gk_last is not None:
        yk = gk - gk_last
        g = (yk.T * sk)[0, 0]
        if (g > 1e-20):
            pk = 1. / g
            t1 = np.eye(args.n_components) - pk * yk * sk.T
            Bk = t1.T * Bk * t1 + pk * sk * sk.T
    
    gk_last = gk
    dk = -Bk * gk
    diag_sum = np.array([abs(Bk[k, k]) for k in range(args.n_components)])
    grad1 = np.matmul(P.transpose(), -dk.A.reshape(-1)) + grad_res * diag_sum.mean() / 10.

    # Backtracking line search
    m = 0
    search_times_MAX = 20

    # Copy the original parameters
    temp_model_name = args.arch + 'temporary.pt'
    torch.save(model.state_dict(), temp_model_name)

    while (m < search_times_MAX):
        sk = rho ** m * dk
        update_grad(model, np.matmul(P.transpose(), -sk.A.reshape(-1)))
        optimizer.step()
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        newf = loss.item()
        model.load_state_dict(torch.load(temp_model_name, map_location="cuda:0"))

        if (newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0,0]):
            # print ('(', m, LA.cond(Bk), ')', end=' ')
            # search_times.append(m)
            break

        m = m + 1
    
    # Cannot find proper lr
    # if m == search_times:
    #     sk *= 0

    # SGD + momentum for the remaining part of gradient
    grad_res_momentum = grad_res_momentum * gamma + grad_res

    # Update the model grad and do a step
    update_grad(model, np.matmul(P.transpose(), -sk.A.reshape(-1)) + grad_res_momentum * alpha)
    optimizer.step()

def validate(val_loader, model, criterion):
    # Run evaluation

    global test_acc, test_loss  

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
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
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # Store the test loss and test accuracy
    test_loss.append(losses.avg)
    test_acc.append(top1.avg)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Save the training model

    torch.save(state, filename)

class AverageMeter(object):
    # Computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    # Computes the precision@k for the specified values of k

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()