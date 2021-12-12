#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading

import time
import datetime

import numpy as np
import shutil
from utils import accuracy, AverageMeter

import sys
from PIL import Image
import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary

import drn
import data_transforms as transforms
from cityscapes_dataset import SegList, SegListMS

from pthflops import count_ops

import lmodels
import pruners.BlockPruner
import pruners.HbPruner
import pruners.RmbPruner
import pruners.RmcdbPruner
import pruners.SRMBRepMasker
import pruners.GroupingPruner

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)

#writer = SummaryWriter("semantic_runs")

# import wandb

# wandb.init(project='semseg_nov', entity='furqan7007')

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        # pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            model.load_state_dict(pretrained_model)
        self.layer = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.layer(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.layer.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

def train(train_loader, model, pruner, criterion, optimizer, epoch, writer, args,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    loss_final = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        #print("Input shape",input.shape)
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            print("here")
            target = target.float()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.long().squeeze(1)
        # compute output
        output = model(input)[0]
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        # print(eval_score)
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Masking if there is a pruner
        if pruner:
            pruner.apply_masks(model)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))
        loss_final = losses.avg
        #print(loss_final)
    print(' * Loss {} : {}'.format(loss_final, epoch))
    writer.add_scalar('Loss/train', loss_final, epoch)
    # wandb.log({'epoch': epoch, 'loss': loss_final})
    

def validate(val_loader, model, criterion, epoch, writer, args, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    # print(model)
    # sys.exit(0)
    model.eval()
    loss_val = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        target_var = target_var.long().squeeze(1)
        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # print(input.size(0))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))
        loss_val = losses.avg
        print(loss_val)
    print(' * Loss {} : {}'.format(loss_val, epoch))
    writer.add_scalar('Loss/Eval', loss_val, epoch)
    # wandb.log({'Val epoch': epoch, ' Val loss': loss_val})
    print(' * Score {top1.avg:.3f}'.format(top1=score))
    # print(' * Loss {loss.avg:.3f}'.format(loss))
    return score.avg


def save_checkpoint(state, is_best, save_dir=".", filename='checkpoint.pth.tar'):
    fpath = os.path.join(save_dir, filename)
    torch.save(state, fpath)
    if is_best:
        shutil.copyfile(fpath, os.path.join(save_dir,'checkpoint_best.pth.tar'))


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print('\t'.join(entries))

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


#def accuracy(output, target, topk=(1,)):
#    """Computes the accuracy over the k top predictions for the specified values of k"""
#    with torch.no_grad():
#        maxk = max(topk)
#        batch_size = target.size(0)

#        _, pred = output.topk(maxk, 1, True, True)
#        pred = pred.t()
#        correct = pred.eq(target.view(1, -1).expand_as(pred))

#        res = []
#        for k in topk:
#            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#            res.append(correct_k.mul_(100.0 / batch_size))
#        return res


#class AverageMeter(object):
#    """Computes and stores the average and current value"""#

#    def __init__(self, window_size=None):
#        self.length = 0
#        self.val = 0
#        self.avg = 0
#        self.sum = 0
#        self.count = 0
#        self.window_size = window_size

#    def reset(self):
#        self.length = 0
#        self.val = 0
#        self.avg = 0
#        self.sum = 0
#        self.count = 0

#    def update(self, val, n=1):
#        if self.window_size and (self.count >= self.window_size):
#            self.reset()
#        self.val = val
#        self.sum += val * n
#        self.count += n
#        self.avg = self.sum / self.count
        
# def accuracy(y_true, y_pred):
#     y_true = y_true.float()
#     _, y_pred = torch.max(y_pred, dim=-1)
#     return (y_pred.float() == y_true).float().mean()
    

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    start_time = time.time()
    for iter, (image, label, name) in enumerate(eval_data_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device {}".format(device))
        image, label = image.to(device), label.to(device)
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        # image_var.cuda()
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        if has_gt:
            label = label.cpu().numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            print('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        print('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    print("Total Inference time {}".format(time.time()-start_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        # input_data.cuda()
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # images.cuda()
        # pdb.set_trace()
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            image_var.cuda()
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        print('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        print(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
        print(single_model)
    # model = torch.nn.DataParallel(single_model).cuda()
    model = single_model.cuda()
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), scales, list_dir=args.list_dir)
    else:
        dataset = SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    # loaded_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            print("loaded epoch {}".format(start_epoch))
            # best_miou1 = checkpoint['best_miou']
            model.load_state_dict(checkpoint['state_dict'])
            # loaded_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # loaded_epoch = checkpoin
    #sparsity = args.pr_config_path.split("_")[15].split("-")[0]
    #sparsity = 0
    # sparsity = args.resume.split("/")[0].split("_")[6]
    # print(sparsity)
    #exit()
    # sparsity = args.sparsity
    out_dir = '{}_ep{:03d}_{}'.format(args.resume.split("/checkpoint")[0], start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt=phase != 'test' or args.with_gt, output_dir=out_dir)
    print('mAP: %f', mAP)


def val_miou(val_loader, model, num_classes, args, has_gt=True):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    start_time = time.time()
    for iter, (image, label) in enumerate(val_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device {}".format(device))
        image, label = image.to(device), label.to(device)
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        # image_var.cuda()
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if has_gt:
            label = label.cpu().numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            print('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        print('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(val_loader), batch_time=batch_time,
                            data_time=data_time))
    print("Total Inference time {}".format(time.time()-start_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)



parser = argparse.ArgumentParser(description='PyTorch Semantic Seg Training')
parser.add_argument("--local_rank", type=int)
parser.add_argument('cmd', choices=['train', 'test', 'calc'])
parser.add_argument('-d', '--data-dir', default=None, required=True)
parser.add_argument('-l', '--list-dir', default=None,
                    help='List dir to look for train_images.txt etc. '
                            'It is the same with --data-dir if not set.')
parser.add_argument('-c', '--classes', default=19, type=int)
parser.add_argument('-s', '--crop-size', default=224, type=int)
parser.add_argument('--step', type=int, default=200)
parser.add_argument('--arch', type = str, default = "drn_d_22")
parser.add_argument('-b', '--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-tb', '--train_batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for Validation (default: 32)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-mode', type=str, default='step')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained',
                    default='', type=str, metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='output path for training checkpoints')
parser.add_argument('--tensorboard_path', default='semseg_runs/', type=str, metavar='PATH',
                    help='path for tensorboard runs')
parser.add_argument('--save_iter', default=1, type=int,
                    help='number of training iterations between'
                            'checkpoint history saves')
parser.add_argument('-j', '--workers', type=int, default=8)
parser.add_argument('--load-release', dest='load_rel', default=None)
parser.add_argument('--phase', default='val')
parser.add_argument('--random-scale', default=4, type=float)
parser.add_argument('--random-rotate', default=0, type=int)
parser.add_argument('--bn-sync', action='store_true')
parser.add_argument('--ms', action='store_true',
                    help='Turn on multi-scale testing')
parser.add_argument('--with-gt', action='store_true')
parser.add_argument('--test-suffix', default='', type=str)
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                 help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    #parser.add_argument("--model", type=str, default="vgg16", help="NN model to use.")
parser.add_argument("--dataset", type=str, default="cityscapes", help="Dataset to use")
parser.add_argument("--exp_dir", type=str, default=".", help="Path to experiment directory", dest="exp_dir")
parser.add_argument("--input_size", type=str, default="1024X768")

# Pruning related
parser.add_argument("--mc_pruning", action="store_true", help="Enable model compression using pruning")
parser.add_argument("--pr-base-model", type=str, help="Path to base dense model", default=None)
parser.add_argument("--pr_config_path", type=str, help="Path to pruning configuration file", default=None)
parser.add_argument("--pr-static", action="store_true", help="Randomly generates structure instead of pruning")
# parser.add_argument("--prune_type", type=str, default=None, help="whether unstructured or structured pruning")

parser.add_argument("--sparsity", type = str, help="Level of sparsity applied")
parser.add_argument("--model", default ='checkpoint_best.pth.tar')

args = parser.parse_args()

assert args.classes > 0


def countNonZeroWeights(model):
    nonzeros = 0
    for param in model.parameters():
        if param is not None:
            nonzeros += torch.sum((param != 0).int()).item()
    return nonzeros



def main():
    
    
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    input_size = args.input_size
    print(f"Input size is {input_size}")
    batch_size = args.train_batch_size
    num_workers = args.workers

    inp_ht = int(input_size.split("X")[0])
    inp_wdth = int(input_size.split("X")[1])
    sparsity = args.sparsity
    num_classes = args.classes
    #print(' '.join(sys.argv))
    # if not os.path.isdir(args.exp_dir):
    #     os.mkdir(args.exp_dir)
    
    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                         pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')
    
    ngpus_per_node = torch.cuda.device_count()
    print(f"Num of gpus per node {ngpus_per_node}")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(single_model)
        model.to(device)
    else:
        model = single_model.cuda()
    # model = single_model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    summary(model, (3,inp_ht,inp_wdth))
    criterion.cuda()    

    pruner = None
    pruner_type = ""
    config_path = args.pr_config_path
    print(f"Config file is {config_path}")
    if args.mc_pruning:
        print("Using "+config_path+" configuration file for generating structure.")

        # import json
        with open(args.pr_config_path) as json_file:
            pruner_type = json.load(json_file)["pruner_type"]
        if pruner_type == "block":
            pruner = pruners.BlockPruner.BlockPruner(config_path)
        elif pruner_type == "hb":
            pruner = pruners.HbPruner.HbPruner(config_path)
        elif pruner_type == "rmb":
            pruner = pruners.RmbPruner.RmbPruner(config_path)
        elif pruner_type == "rmcdb":
            pruner = pruners.RmcdbPruner.RmcdbPruner(config_path)
        elif pruner_type == "grouping":
            pruner = pruners.GroupingPruner.GroupingPruner(config_path)
        elif pruner_type == "srmbrep":
            pruner = pruners.SRMBRepMasker.SRMBRepMasker(config_path)
        else:
            print("Unsupported pruner ", pruner_type)
            exit(-1)

        # Pruning related
        pruner.generate_masks(model, is_static=args.pr_static, verbose=True)
        pruner.print_stats()

    lr = args.lr
    # Learning rate scheduler
    if args.epochs == 400: #imagenet
        ## WRN configuration
        milestones = [60,120,180,240,300,350]
        gamma = 0.1
    elif args.epochs == 500: #imagenet
        ## WRN configuration
        milestones = [60,120,180,240,300,350,400,450]
        gamma = 0.1
    if args.epochs == 250: #imagenet
        ## WRN configuration
        milestones = [50,100,150,200]
        gamma = 0.1
    elif args.epochs == 300:
        milestones = [50,100,150,200,250]
        gamma = 0.1
    elif args.epochs == 200:
        milestones = [40,80,120,160]
        gamma = 0.1
    elif args.epochs == 150:
        milestones = [40,80,120]
        gamma = 0.1
    
    if args.cmd == 'train':
        # Data loading code
        print("in train")
        data_dir = args.data_dir
        info = json.load(open(join(data_dir, 'info.json'), 'r'))
        normalize = transforms.Normalize(mean=info['mean'],
                                        std=info['std'])
        t = []

        t.extend([transforms.RandomCrop((inp_ht,inp_wdth)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Resize(255),
                normalize])

        # sparsity = args.pr_config_path.split("/")[2].split("-")[0]
        print("Sparsity {}".format(sparsity))
        sparsity = float(sparsity)
        if sparsity > 0 and pruner_type == "srmbrep":
            osp = str(args.pr_config_path.split("/")[2].split("_")[11].split("-")[0])
            isp = str(args.pr_config_path.split("/")[2].split("_")[12].split("-")[0])
            obh_size = str(args.pr_config_path.split("/")[2].split("_")[8:11][0])
            cbh_size = str(args.pr_config_path.split("/")[2].split("_")[8:11][1])
            ibh_size = str(args.pr_config_path.split("/")[2].split("_")[8:11][2])
            print(f" isp {isp},\n osp {osp},\n obh_size {obh_size},\n cbh_size {cbh_size},\n ibh_size {ibh_size} \n")
        elif pruner_type == "block":
            pass
        else:
            isp = 0
            osp = 0
            obh_size = 0
            cbh_size = 0
            ibh_size = 0
            print(f" isp {isp},\n osp {osp},\n obh_size {obh_size},\n cbh_size {cbh_size},\n ibh_size {ibh_size} \n")
        if args.dataset == "cityscapes":
            train_loader = torch.utils.data.DataLoader(
            SegList(data_dir, 'train', transforms.Compose(t),
                    list_dir=args.list_dir),
            batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                SegList(data_dir, 'val', transforms.Compose([
                    transforms.RandomCrop((inp_ht, inp_wdth)),
                    transforms.ToTensor(),
                    normalize,
                ]), list_dir=args.list_dir),
                batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=True, drop_last=True
            )
        elif args.dataset == "ade20k":
            train_loader = torch.utils.data.DataLoader(
            ADEDataset(data_dir, 'training', transforms.Compose(t),
                    list_dir=args.list_dir),
            batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                ADEDataset(data_dir, 'validation', transforms.Compose([
                    transforms.RandomCrop((inp_ht, inp_wdth)),
                    transforms.ToTensor(),
                    normalize,
                ]), list_dir=args.list_dir),
                batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=True, drop_last=True
            )

        elif args.dataset == "voc":
            NUM_CHANNELS = 3
            NUM_CLASSES = 22
            color_transform = Colorize()
            image_transform = ToPILImage()
            input_transform = Compose([
                CenterCrop(256),
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
            target_transform = Compose([
                CenterCrop(256),
                ToLabel(),
                Relabel(255, 21),
            ])

            
            train_loader = torch.utils.data.DataLoader(VOC12(data_dir, input_transform, target_transform),
            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
        # define loss function (criterion) and pptimizer
        optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        cudnn.benchmark = True
        best_prec1 = 0
        start_epoch = 0
        best_miou = 0
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                #best_prec1 = checkpoint['best_prec1']
                best_miou = checkpoint['best_miou']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                # writer = SummaryWriter(args.tensorboard_path+"/"+args.arch+"res_"+args.resume.split("/")[0].split("_")[3].split(".")[0]+"ep"+str(start_epoch)+"_b"+str(args.batch_size)+"_e"+str(args.epochs)+"_lr"+str(args.lr)+"date"+str(datetime.date.today()))
                # save_path = args.arch+"res_"+args.resume.split("/")[0].split("_")[3].split(".")[0]+"ep"+str(start_epoch)+"_b"+str(args.batch_size)+"_e"+str(args.epochs)+"_lr"+str(args.lr)+"date"+str(datetime.date.today())
                # print(os.path)
                # if not os.path.exists(save_path):
                #     print("making dir")
                #     os.makedirs(save_path)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                exit()

        if args.epochs == 100:
            if args.resume:
                milestones = [start_epoch+25,start_epoch+50, start_epoch+75]
                gamma = 0.1

        elif args.epochs == 150:
            if args.resume:
                milestones = [start_epoch+40,start_epoch+80,start_epoch+120]
                gamma = 0.1
            
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        if args.evaluate:
            validate(val_loader, model, criterion, eval_score=accuracy)
            return
        
        if not args.resume:
            if pruner_type == "block":
                save_path = args.pr_config_path.split("/config")[0]+"_lr"+str(args.lr)+"_e"+str(args.epochs)
                writer = SummaryWriter(args.tensorboard_path+"/"+args.arch+"_inp_size"+str(input_size)+"_e"+str(args.epochs)+"_b"+str(args.batch_size)+"_lr"+str(args.lr)+"_date"+str(datetime.date.today())+"time"+str(time.time()))
                # save_path = args.arch+"_e"+str(args.epochs)+"_b"+str(args.batch_size)+"_lr"+str(args.lr)+"_sp"+str(sparsity)+"_"+obh_size+"_"+cbh_size+"_"+ibh_size+"_date"+str(datetime.date.today())+"time"+str(time.time())
                if not os.path.exists(save_path):
                    print("making a path")
                    os.makedirs(save_path)
            elif pruner_type == "srmbrep":
                save_path = args.pr_config_path.split("/config")[0]+"_lr"+str(args.lr)+"_e"+str(args.epochs)
                writer = SummaryWriter(args.tensorboard_path+"/"+args.arch+"_inp_size"+str(input_size)+"_e"+str(args.epochs)+"_b"+str(args.batch_size)+"_lr"+str(args.lr)+"_isp"+str(isp)+"_osp"+str(osp)+"_"+obh_size+"_"+cbh_size+"_"+ibh_size+"_date"+str(datetime.date.today())+"time"+str(time.time()))
                # save_path = args.arch+"_e"+str(args.epochs)+"_b"+str(args.batch_size)+"_lr"+str(args.lr)+"_sp"+str(sparsity)+"_"+obh_size+"_"+cbh_size+"_"+ibh_size+"_date"+str(datetime.date.today())+"time"+str(time.time())
                if not os.path.exists(save_path):
                    print("making a path")
                    os.makedirs(save_path)
        else:
            save_path = args.pr_config_path.split("/config")[0]+"_resume"+"_"+str(start_epoch)+"_till"+str(args.epochs)+"_lr"+str(args.lr)
            writer = SummaryWriter(args.tensorboard_path+"/"+args.arch+"_resume"+"_"+str(start_epoch)+"_b"+str(args.batch_size)+"_inp_size"+str(input_size)+"_lr"+str(args.lr)+"_isp"+str(isp)+"_osp"+str(osp)+"_"+obh_size+"_"+cbh_size+"_"+ibh_size+"_date"+str(datetime.date.today())+"time"+str(time.time())+"_e"+"_"+str(args.epochs))
            # save_path = args.arch+"_resume"+"_"+str(start_epoch)+"_b"+str(args.batch_size)+"_lr"+str(args.lr)+"_sp"+str(sparsity)+"_"+obh_size+"_"+cbh_size+"_"+ibh_size+"_date"+str(datetime.date.today())+"time"+str(time.time())+"_e"+str(args.epochs)
            if not os.path.exists(save_path):
                print("making a path")
                os.makedirs(save_path)
            
            
        # Rejig initialization
        if args.mc_pruning and args.pr_static:
            with torch.no_grad():
                for layer in pruner.mask_dict:
                    # Get tensor and mask
                    tensor =  model.state_dict()[layer]
                    mask   = pruner.mask_dict[layer]

                    # Applying mask
                    tensor.mul_(mask)

                    # Create small tensor
                    nnz_count = torch.sum(mask != 0).item()
                    n = nnz_count//mask.shape[1]
                    small_tensor = torch.zeros(nnz_count, dtype=tensor.dtype,
                                                layout=tensor.layout, device=tensor.device)

                    if len(tensor.shape) == 2:
                        print("Reinitializing FC {} wrt sparsity".format(layer))
                        small_tensor.normal_(0, 0.01)
                    else:
                        print("Reinitializing CONV {} wrt sparsity".format(layer))
                        small_tensor.normal_(0, math.sqrt(2. / n))

                    # Distribute the values to big tensor
                    tensor[torch.nonzero(mask, as_tuple=True)] = small_tensor.flatten()

        if args.mc_pruning:
            # if not args.pr_static:
            #     print("Base line dense accuracy")
            #     validate(val_loader, model, criterion, args)
            print("Applying masking before training begins")
            pruner.apply_masks(model)

        if args.resume:
            best_epoch = start_epoch
        else:
            best_epoch = 0
        print("Start epoch {}".format(start_epoch))
        for epoch in range(start_epoch,start_epoch+args.epochs):

            train(train_loader, model, pruner, criterion, optimizer, epoch, writer, args, eval_score=accuracy)

            miou1 = val_miou(val_loader, model, num_classes, args)
            print(" Val miou {} : {}".format(miou1, epoch))
            writer.add_scalar("Miou/Val", miou1, epoch)
            # writer.add_scalar('Loss/Eval', loss_val, epoch)
            # wandb.log({'Val epoch': epoch, ' Val miou': miou1})
            is_best_miou = miou1 > best_miou
            if is_best_miou:
                best_miou = max(miou1,best_miou)
                best_epoch = epoch
            

            save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_miou': best_miou,
            'optimizer' : optimizer.state_dict(),
            'dataset' : args.dataset
            }, is_best_miou, save_path)

            if pruner:
                pruner.print_stats()

            print("Best MioU till epoch:{} : {}".format(best_epoch, best_miou))
            # print("Best Top-1 error : till epoch {} : {}".format(epoch, 100 - best_prec1))


    elif args.cmd == 'test':
        test_seg(args)

    elif args.cmd == 'calc':
        #Calculate the number of non zero weights
        checkpoint = torch.load(args.model)
        #        start_epoch = checkpoint['epoch']
        #        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        nonzeros = countNonZeroWeights(model)

        print("Num of non zero weights in the model {}".format(nonzeros))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inp = torch.rand(batch_size,3,256,512).to(device)

        # Count the number of FLOPs
        count_ops(model, inp)  



if __name__ == '__main__':
    main()
