import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # --- ReLU-Conv-BN-1 ---
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # --- ReLU-Conv-BN-2 ---
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # --- residual conn ---
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        # --- ReLU-Conv-BN-1 ---
        identity = out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        # --- ReLU-Conv-BN-2 ---
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # --- residual conn ---
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # --- ReLU-Conv-BN-1 ---
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # --- ReLU-Conv-BN-2 ---
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # --- ReLU-Conv-BN-3 ---
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # --- residual conn ---
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        # --- ReLU-Conv-BN-1 ---
        identity = out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        # --- ReLU-Conv-BN-2 ---
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # --- ReLU-Conv-BN-3 ---
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # --- residual conn ---
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        return out


# TODO: dropout
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, data_dim=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # --- stem ---
        self.conv1 = nn.Conv2d(data_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

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


def resnet18(**kwargs):
    return ResNet(ResNetBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(ResNetBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(ResNetBottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(ResNetBottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(ResNetBottleneck, [3, 8, 36, 3], **kwargs)
