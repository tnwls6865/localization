import argparse
import csv
import os
import shutil
import time
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataloader import Cub2011
import resnet as RN
import utils

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--net_type', default='resnet', type=str, help='networktype: resnet')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=50, type=int, help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='cub', type=str, help='dataset (options: cifar10, cifar100, and tiny)')
parser.add_argument('--expname', default='base', type=str, help='name of experiment')
parser.add_argument('--beta', default=0.0, type=float, help='hyperparameter beta')
parser.add_argument('--mix_prob', default=0.0, type=float, help='mix probability')
parser.add_argument('--resume', type=bool, default=False, help='restart')                   

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_acc1 = 0
best_acc5 = 0
cudnn.benchmark = False

def main():
    global args, start_epoch
    args = parser.parse_args()
    test_id = args.net_type  + str(args.depth) + '_' + args.expname + '_CAM'
    print('test_id : ', test_id)

    csv_path = os.path.join('logs', test_id)
    checkpoint_path = os.path.join('save_models', test_id)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
            
    print('csv_path : ', csv_path)
    print('models_path : ', checkpoint_path)


    # Preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Dataloader 
    train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize])

    val_transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.CenterCrop((224)),
                                        transforms.ToTensor(),
                                        normalize])

    train_dataset = Cub2011(root='../datas/', train=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = Cub2011(root='../datas/', train=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    numberofclass = 200

    model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet

    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss().cuda()
    
    start_epoch = 0
    filename = csv_path + '/' + test_id + '.csv'
    csv_logger = utils.CSVLogger(csv_path, args=args, fieldnames=['epoch', 'train_loss', 'train_acc1', 'test_loss', 'test_acc1', 'test_acc5'], filename=filename)

    if args.resume:
        checkpoint_path = checkpoint_path + csv_path
        checkpoint = torch.load(checkpoint_path + 'model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        filename = csv_path + 'log.csv'
        csv_logger = utils.CSVLogger(csv_path, args=args, fieldnames=['epoch', 'train_loss', 'train_acc1', 'test_loss', 'test_acc1', 'test_acc5'], filename=filename)
        
    for epoch in range(start_epoch, start_epoch + args.epochs):

            utils.adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            train_acc1, train_loss = train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)
            
            # loss
            train_loss = '%.4f' % (train_loss)
            train_acc1 = '%.4f' % (train_acc1)

            val_loss = '%.4f' % (val_loss)
            test_acc1 = '%.4f' % (acc1)
            test_acc5 = '%.4f' % (acc5)

            # remember best prec@1 and save checkpoint
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_acc5 = acc5

            print('Current best acc (top-1 {0:.3f} and 5 acc {1:.3f})'.format(best_acc1, best_acc5))
            print(' ')
            utils.save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }, is_best, test_id)

            row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'train_acc1': str(train_acc1), 'test_loss': str(val_loss), 'test_acc1': str(test_acc1), 'test_acc5': str(test_acc5)}
            csv_logger.writerow(row)

    print('Best accuracy (top-1 {0:.3f} and 5 acc {1:.3f})'.format(best_acc1, best_acc5))
    csv_logger.close()

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = utils.get_learning_rate(optimizer)[0]
    for i, (imgs, labels, bbox) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        imgs = imgs.cuda()
        labels = labels.cuda()

        # r = np.random.rand(1)
        if args.beta > 0 and args.mix_prob > 0:
            # generate mixed sample
            is_train = True
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(imgs.size()[0]).cuda()
            
            labels_a = labels
            labels_b = labels[rand_index]

            bbx1, bby1, bbx2, bby2 = utils.rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
            
            # compute output
            output = model(imgs, is_train, rand_index, lam)
            loss = criterion(output, labels_a) * lam + criterion(output, labels_b) * (1. - lam)
        else:
            # compute output
            is_train = False
        
            output = model(imgs, is_train)
            loss = criterion(output, labels)

        # measure accuracy and record loss
        # _, preds = torch.max(output.data, 1)
        err1, err5 = utils.accuracy(output.data, labels, topk=(1, 5))

        losses.update(loss.item(), imgs.size(0))
        top1.update(err1.item(), imgs.size(0))
        top5.update(err5.item(), imgs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs + start_epoch, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Train Loss {loss.avg:.3f} \n'.format(
        epoch, args.epochs + start_epoch, top1=top1, top5=top5, loss=losses))

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, epoch):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    is_train = False

    end = time.time()
    for i, (imgs, labels, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            output = model(imgs, is_train)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        err1, err5 = utils.accuracy(output.data, labels, topk=(1, 5))

        losses.update(loss.item(), imgs.size(0))

        top1.update(err1.item(), imgs.size(0))
        top5.update(err5.item(), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs + start_epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs + start_epoch, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    main()