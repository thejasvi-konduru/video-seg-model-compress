import torch
import sys
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import os
import numpy as np
import math
# Loading torch models and local models
import torchvision.models as models
sys.path.insert(0, '../lmodels')
from lmodels import drn
import lmodels

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


def get_train_and_val_loaders(args):
    # Data loaders
    train_sampler = None
    if args.dataset == "imagenet":
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

    elif args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])


        train_set = datasets.CIFAR10(root='./data/cifar10/', train=True,
                            download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root="./data/cifar10/", train=False,
                            download=True, transform=transform_test)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == "cifar100":
        normalize = transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]],
                std=[n/255. for n in [68.2,  65.4,  70.4]])
        #normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
        #		std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])


        train_set = datasets.CIFAR100(root='./data/cifar100/', train=True,
                            download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root="./data/cifar100/", train=False,
                            download=True, transform=transform_test)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == "mnist":

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        train_set = datasets.MNIST(root='./data/mnist/', train=True,
                            download=True, transform=transform)
        test_set = datasets.MNIST(root="./data/mnist/", train=False,
                            download=True, transform=transform)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)


        val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        print("Not a valid dataset", args.dataset)
        sys.exit(-1)

    return train_loader, val_loader, train_sampler


def create_model(dataset, arch, pretrained=False):

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    lmodel_names = sorted(name for name in lmodels.__dict__
        if name.islower() and not name.startswith("__")
        and callable(lmodels.__dict__[name]))

    # Fixing number of classes
    if dataset == "imagenet":
        num_classes = 1000
    elif dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "cityscapes":
        num_classes = 19
    else:
        print("Invalid dataset")
        exit(-1)

    # create model
    if arch in model_names:
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](num_classes = num_classes, pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch](num_classes = num_classes)
    elif arch in lmodel_names:
        model = lmodels.__dict__[arch](num_classes=num_classes)
    elif arch in ["drn_d_22", "drn_d_38"]:
        model = drn.__dict__.get(arch)(pretrained=pretrained, num_classes=num_classes)
        # pmodel = nn.DataParallel(model)
        # if pretrained_model is not None:
        #     pmodel.load_state_dict(pretrained_model, strict=False)
        base = nn.Sequential(*list(model.children())[:-2])

        seg = nn.Conv2d(model.out_dim, num_classes,
                             kernel_size=1, bias=True)
        softmax = nn.LogSoftmax()
        m = seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        # if use_torch_up:
        #     up = nn.UpsamplingBilinear2d(scale_factor=8)
        # else:
        up = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=num_classes,
                                    bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        #    up = up
    else:
        print("Invalid model name ", arch)
        exit(-1)

    return model

def get_model_information(dataset, arch):
    import collections
    flop_dict = collections.OrderedDict()
    param_dict = collections.OrderedDict()
    parent_lists = collections.OrderedDict()
    child_lists = collections.OrderedDict()

    if "cifar" in dataset:
        if arch == "cifar_resnet18":
            json_fp = "cifar_resnet18.json"
        if arch == "cifar_vgg16_bn":
            json_fp = "cifar_vgg16_bn.json"


    # Reading from json
    import json
    with open(json_fp) as json_file:
        data = json.load(json_file)
        for layer in data:
            linfo = data[layer]
            filter_size = (linfo["ifm"] * linfo["ks"][0] * linfo["ks"][1])
            lflops = (linfo["ofm"] * linfo["oh"] * linfo["ow"]) * filter_size
            flop_dict[layer] = lflops

            lparams = linfo["ofm"] * linfo["ifm"] * linfo["ks"][0] * linfo["ks"][1]
            param_dict[layer] = lparams

            parent_lists[layer] = linfo["parents"]
            child_lists[layer]  = linfo["children"]


    return param_dict, flop_dict, parent_lists, child_lists


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data