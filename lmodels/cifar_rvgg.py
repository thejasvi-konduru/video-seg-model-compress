import torch.nn as nn


__all__ = [
    'Cifar_RVGG',
    'cifar_res_rvgg11_64_bn', 'cifar_res_rvgg11_128_bn', 'cifar_res_rvgg11_256_bn', 'cifar_res_rvgg11_512_bn',
]


class Cifar_RVGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, fmaps=512):
        super(Cifar_RVGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(fmaps, fmaps),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fmaps, fmaps),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fmaps, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class BasicBlock(nn.Module):
    """docstring for BasicBlock"""
    def __init__(self, in_planes, planes, kernel_size, padding, batch_norm, add_res):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm
        self.add_res = add_res
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding)
        if self.batch_norm:
            self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)

        if self.add_res:
            assert(in_planes == planes)


    def forward(self, x):
        identity = x

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)

        if self.add_res:
            out += identity

        out = self.relu(out)

        return out
        


def make_layers(cfg, batch_norm=False, add_res=False):
    layers = []
    in_channels = 3
    conv2d = nn.Conv2d(in_channels, cfg[0], kernel_size=3, padding=1)
    if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(cfg[0]), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]

    # Adjusting cfg
    in_channels = cfg[0]
    cfg = cfg[1:]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            res_conv2d = BasicBlock(in_channels, v, kernel_size=3, padding=1, batch_norm=batch_norm, add_res=add_res)
            layers += [res_conv2d]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'R64':[64,'M', 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M'],
    'R128':[128,'M', 128, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'],
    'R256':[256,'M', 256, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256, 'M'],
    'R512':[512,'M', 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, add_res, **kwargs):
    model = Cifar_RVGG(make_layers(cfgs[cfg], batch_norm=batch_norm, add_res=add_res), **kwargs)
    return model


def cifar_rvgg11_64_bn(**kwargs):
    return _vgg('R64', True, False, fmaps=64, **kwargs)

def cifar_rvgg11_128_bn(**kwargs):
    return _vgg('R128', True, False, fmaps=128, **kwargs)

def cifar_rvgg11_256_bn(**kwargs):
    return _vgg('R256', True, False, fmaps=256, **kwargs)

def cifar_rvgg11_512_bn(**kwargs):
    return _vgg('R512', True, False, fmaps=512, **kwargs)


def cifar_res_rvgg11_64_bn(**kwargs):
    return _vgg('R64', True, True, fmaps=64, **kwargs)

def cifar_res_rvgg11_128_bn(**kwargs):
    return _vgg('R128', True, True, fmaps=128, **kwargs)

def cifar_res_rvgg11_256_bn(**kwargs):
    return _vgg('R256', True, True, fmaps=256, **kwargs)

def cifar_res_rvgg11_512_bn(**kwargs):
    return _vgg('R512', True, True, fmaps=512, **kwargs)

"""
if __name__ == "__main__":
    model =  cifar_res_rvgg11_512_bn()
    print(model)
"""
