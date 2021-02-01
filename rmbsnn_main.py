import argparse
import os
import random
import shutil
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import utils

import lmodels
import pruners.BlockPruner
import pruners.HbPruner
import pruners.RmbPruner
import pruners.RmcdbPruner
import pruners.SRMBRepMasker
import pruners.GroupingPruner

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

lmodel_names = sorted(name for name in lmodels.__dict__
    if name.islower() and not name.startswith("__")
    and callable(lmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names + lmodel_names,
                    help='model architecture: ' +
                        ' | '.join(model_names + lmodel_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
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
parser.add_argument("--dataset", type=str, default="imagenet", 
                    help="Dataset to use")
parser.add_argument("--exp-dir", type=str, default="experiments/",
                    help="Path to experiment directory", dest="exp_dir")

# Pruning related
parser.add_argument("--mc-pruning", action="store_false", help="Enable model compression using pruning")
parser.add_argument("--pr-base-model", type=str, help="Path to base dense model", default=None)
parser.add_argument("--pr-config-path", type=str, help="Path to pruning configuration file", default="sample_configs/imagenet_resnet50_hb_pconfig.json")
parser.add_argument("--pr-static", action="store_false", help="Randomly generates structure instead of pruning")

# Knowledge distillation
parser.add_argument("--mc-kd", action="store_true", help="Enable model compression using knowledge distillation")
parser.add_argument("--kd-teacher", type=str, help="Teacher model")
parser.add_argument("--kd-temperature", type=float, default=1, help="Softmax with temperature to create soft labels")
parser.add_argument("--kd-student-wt", type=float, default=0.5, help="Weighting parameter for student & hard labels")
parser.add_argument("--kd-distill-wt", type=float, default=0.5, help="Weighting parameter for student & soft labels")


best_acc1 = 0


def main():
    args = parser.parse_args()

    # Create experiment directory if does not exists
    import os
    if not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("Num of gpus per node",ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    

    # Fixing number of classes
    model = utils.create_model(args.dataset, args.arch, args.pretrained)

    #### Pruning related code ####
    if args.mc_pruning and not args.pr_static:
        print("Setting network to a base dense model.")
        # Over riding values
        checkpoint = torch.load(args.pr_base_model)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        for key in checkpoint:
            mkey = key
            if "module" in key:
                mkey = key.replace("module.","")
            
            # Copying the
            model.state_dict()[mkey].copy_(checkpoint[key])

    #### Knowledge distillation related code ####
    teacher_model = None
    if args.mc_kd:
        teacher_model = utils.create_model(args.dataset, args.arch, args.pretrained)

        print("Setting teacher network")
        # Over riding values
        checkpoint = torch.load(args.kd_teacher)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        for key in checkpoint:
            mkey = key
            if "module" in key:
                mkey = key.replace("module.","")
            
            # Copying the teacher model
            teacher_model.state_dict()[mkey].copy_(checkpoint[key])

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    #### Knowledge distillation related code ####
    if args.mc_kd:
        # ASSUMPTION : args.distributed is False, args.gpu is None
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()

    #### Pruning related code ####
    pruner = None
    if args.mc_pruning:
        print("Using "+args.pr_config_path+" configuration file for generating structure.")

        import json
        with open(args.pr_config_path) as json_file:
            pruner_type = json.load(json_file)["pruner_type"]
        if pruner_type == "block":
            pruner = pruners.BlockPruner.BlockPruner(args.pr_config_path)
        elif pruner_type == "hb":
            pruner = pruners.HbPruner.HbPruner(args.pr_config_path)
        elif pruner_type == "rmb":
            pruner = pruners.RmbPruner.RmbPruner(args.pr_config_path)
        elif pruner_type == "rmcdb":
            pruner = pruners.RmcdbPruner.RmcdbPruner(args.pr_config_path)
        elif pruner_type == "grouping":
            pruner = pruners.GroupingPruner.GroupingPruner(args.pr_config_path)
        elif pruner_type == "srmbrep": 
            pruner = pruners.SRMBRepMasker.SRMBRepMasker(args.pr_config_path)
        else:
            print("Unsupported pruner ", pruner_type)
            exit(-1)

        # Pruning related
        pruner.generate_masks(model, is_static=args.pr_static, verbose=True)
        pruner.print_stats()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Learning rate scheduler
    if args.epochs == 200:
        ## WRN configuration
        milestones = [60,120,160]
        gamma = 0.2
    elif args.epochs == 160:
        milestones = [80,120]
        gamma = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    """
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    """

    # Data loading code
    train_loader, val_loader, train_sampler = utils.get_train_and_val_loaders(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

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
                # tensor[torch.nonzero(mask, as_tuple=True)] = small_tensor.flatten()
                tensor[torch.nonzero(mask, as_tuple=True)] = small_tensor.flatten()


    if args.mc_pruning:
        if not args.pr_static:
            print("Base line dense accuracy")
            validate(val_loader, model, criterion, args)
    
        print("Applying masking before training begins")
        pruner.apply_masks(model)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, pruner, teacher_model)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # Learning rate step
        lr_scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'dataset' : args.dataset
            }, is_best, args.exp_dir)

        if pruner:
            pruner.print_stats()

    print("Best Top-1 accuracy : ", best_acc1)
    print("Best Top-1 error : ", 100 - best_acc1)


def train(train_loader, model, criterion, optimizer, epoch, args, pruner=None, teacher_model=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(images)
                
            import torch.nn.functional as F
            soft_log_probs = F.log_softmax(output / args.kd_temperature, dim=1)
            soft_targets = F.softmax(teacher_output / args.kd_temperature, dim=1)

            distillation_loss = F.kl_div(soft_log_probs, soft_targets.detach(), size_average=False) / soft_targets.shape[0]

            loss = args.kd_student_wt * loss + args.kd_distill_wt * distillation_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

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

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, exp_dir='.', filename='checkpoint.pth.tar'):
    fpath = os.path.join(exp_dir, filename)
    torch.save(state, fpath)
    if is_best:
        shutil.copyfile(fpath, os.path.join(exp_dir,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset == "imagenet":
        if args.epochs == 100:
            lr = args.lr * (0.1 ** (epoch // 30))
        else:
            portions = [1/2, 3/4]
            milestones = [int(portion*args.epochs) for portion in portions]

            portion_id = 0
            for milestone in milestones:
                if epoch >= milestone:
                    portion_id += 1
                else:
                    break

            lr = args.lr * pow(0.1, portion_id)

        # Weight parameters
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        """
        portions = [1/2, 3/4]
        if args.epochs in [120,240]:
            portions = [3/6, 4/6, 5/6]
        milestones = [int(portion*args.epochs) for portion in portions]

        portion_id = 0
        for milestone in milestones:
            if epoch >= milestone:
                portion_id += 1
            else:
                break

        lr = args.lr * pow(0.1, portion_id)
        """

    


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
