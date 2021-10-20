import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def min_max(input):
    min= torch.min(input)
    max= torch.max(input)
    output = (input - min) / (max - min)
    return output

class BasicBlock(nn.Module):
    expansion = 1
    channel_num = 128

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    channel_num = 512

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #feture extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)
        #attention branch
        self.att_conv4_good  = nn.Conv2d(block.channel_num*2, block.channel_num*2, kernel_size=3, padding=1, bias=False)
        self.att_conv6_good  = nn.Conv2d(block.channel_num*2, block.channel_num, kernel_size=3, padding=1, bias=False)
        self.att_conv4_bad  = nn.Conv2d(block.channel_num*2, block.channel_num*2, kernel_size=3, padding=1, bias=False)
        self.att_conv6_bad  = nn.Conv2d(block.channel_num*2, block.channel_num, kernel_size=3, padding=1, bias=False)
        self.bn_att4 = nn.BatchNorm2d(block.channel_num*2)
        self.bn_att6 = nn.BatchNorm2d(block.channel_num)
        self.att_wgp_good = nn.Conv2d(block.channel_num, 1,(14,14),padding=0, bias=False)
        self.att_wgp_bad = nn.Conv2d(block.channel_num, 1,(14,14),padding=0, bias=False)
        #perception branch
        self.Tanh = nn.Tanh()
        self.layer4_good = self._make_layer(block, 512, layers[3], stride=2, down_size=False)
        self.layer4_bad = self._make_layer(block, 512, layers[3], stride=2, down_size=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.new_fc_good = nn.Linear(512 * block.expansion, 1)
        self.new_fc_bad = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fe = x

        #####superior_attention_branch
        ax_good = self.relu(self.bn_att4(self.att_conv4_good(x)))
        ax_good = self.relu(self.bn_att6(self.att_conv6_good(ax_good)))
        att_map_good = torch.sum(ax_good, dim=1,keepdim = True)
        self.att_good = min_max(att_map_good)
        ax_good = self.att_wgp_good(ax_good)

        #####inferior_attention_branch
        ax_bad = self.relu(self.bn_att4(self.att_conv4_bad(x)))
        ax_bad = self.relu(self.bn_att6(self.att_conv6_bad(ax_bad)))
        att_map_bad = torch.sum(ax_bad, dim=1,keepdim = True)
        self.att_bad = min_max(att_map_bad)
        ax_bad = self.att_wgp_bad(ax_bad)

        #####change
        rx_good = x * self.att_good
        rx_bad = x * self.att_bad
        #rx_bad = x * self.att_good
        #rx_good = x * self.att_bad

        #####superior_perception_branch
        per_good = rx_good
        rx_good = self.layer4_good(rx_good)
        rx_good = self.avgpool(rx_good)
        rx_good = rx_good.view(rx_good.size(0), -1)
        rx_good = self.new_fc_good(rx_good)
        rx_good = self.Tanh(rx_good)

        #####inferior_perception_branch
        per_bad = rx_bad
        rx_bad = self.layer4_bad(rx_bad)
        rx_bad = self.avgpool(rx_bad)
        rx_bad = rx_bad.view(rx_bad.size(0), -1)
        rx_bad = self.new_fc_bad(rx_bad)
        rx_bad = self.Tanh(rx_bad)

        return ax_good, ax_bad, rx_good, rx_bad, [self.att_good, self.att_bad,fe, per_good, per_bad]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model