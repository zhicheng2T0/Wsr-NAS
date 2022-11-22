import torch
import torch.nn as nn

from ..resnet import ResNetBlock, ResNetBottleneck


class ResNetImageNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, data_dim=3):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64

        # --- stem ---
        self.conv1 = nn.Conv2d(data_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- layers ---
        # layer 1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # layer 2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # layer 3
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # layer 4
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # --- global average pooling and classifier ---
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- stem ---
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # --- layers ---
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # --- global average pooling and classifier ---
        out = self.relu(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18_imagenet(**kwargs):
    return ResNetImageNet(ResNetBlock, [2, 2, 2, 2], **kwargs)

def resnet34_imagenet(**kwargs):
    return ResNetImageNet(ResNetBlock, [3, 4, 6, 3], **kwargs)

def resnet50_imagenet(**kwargs):
    return ResNetImageNet(ResNetBottleneck, [3, 4, 6, 3], **kwargs)

def resnet101_imagenet(**kwargs):
    return ResNetImageNet(ResNetBottleneck, [3, 4, 23, 3], **kwargs)

def resnet152_imagenet(**kwargs):
    return ResNetImageNet(ResNetBottleneck, [3, 8, 36, 3], **kwargs)
