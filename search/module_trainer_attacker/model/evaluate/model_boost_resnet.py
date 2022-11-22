import logging

import torch
import torch.nn as nn

from module.model.evaluate import BoostNetwork, EvaluateBlock, Cell
from module.model.backbone import ResNetBlock, ResNetBottleneck


class BottleneckBlock(nn.Module):

    def __init__(self, genotype, num_layer, C_prev_prev, C_prev, C,
                 expansion, reduction, reduction_prev):
        super(BottleneckBlock, self).__init__()
        self.num_layer = num_layer
        self.reduction = reduction
        self.expansion = expansion

        # down sample
        self.down_1 = nn.Sequential(
            nn.Conv2d(C_prev_prev, C_prev_prev // self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_prev_prev // self.expansion)
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(C_prev, C_prev // self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_prev // self.expansion)
        )

        # regular block
        layers = [Cell(genotype=genotype,
                       C_prev_prev=C_prev_prev // self.expansion,
                       C_prev=C_prev // self.expansion,
                       C=C // self.expansion,
                       reduction=reduction, reduction_prev=reduction_prev)]
        C_prev_prev, C_prev = C_prev, C * layers[-1].multiplier
        for _ in range(1, num_layer):
            layers.append(
                Cell(genotype=genotype,
                     C_prev_prev=C_prev_prev // self.expansion,
                     C_prev=C_prev // self.expansion,
                     C=C // self.expansion,
                     reduction=False, reduction_prev=reduction)
            )
            reduction = False
            C_prev_prev, C_prev = C_prev, C * layers[-1].multiplier

        # up sample
        self.up = nn.Sequential(
            nn.Conv2d(C // self.expansion * layers[-1].multiplier,
                      C * layers[-1].multiplier,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(C * layers[-1].multiplier)
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, s0, s1, drop_prob):

        out = None

        s0 = self.down_1(s0)
        s1 = self.down_2(s1)

        for layer in self.layers:
            # print(s0.size(), s1.size())
            out = layer(s0, s1, drop_prob=drop_prob)
            s0, s1 = s1, out

        out = self.up(out)

        return out

    def cal_flops(self):
        flops = 0.
        flops += self.layers[0].cal_flops()
        stride = 2 if self.reduction else 1
        for layer in self.layers[1:]:
            flops += layer.cal_flops() / stride / stride
        return flops


class ResNet(BoostNetwork):
    def __init__(self, genotype, block, num_blocks, data_dim=3,
                 num_classes=10, weak_layers=12, drop_prob=0.):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.genotype = genotype
        self.multiplier = len(self.genotype.ops_concat)
        assert weak_layers % 4 == 0
        self.weak_layers = weak_layers // 4
        self.drop_prob = drop_prob

        # --- stem ---
        self.conv1 = nn.Conv2d(data_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # --- layers ---
        # layer 1
        c_in = c_in_prev = self.in_planes
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.weak1 = self._make_weak_learner(c_in, c_in_prev, c_out=self.in_planes,
                                             expansion=1,
                                             reduction=False, reduction_prev=False)
        # layer 2
        c_in, c_in_prev = self.in_planes, c_in
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.weak2 = self._make_weak_learner(c_in, c_in_prev, c_out=self.in_planes,
                                             expansion=block.expansion,
                                             reduction=True, reduction_prev=False)
        # layer 3
        c_in, c_in_prev = self.in_planes, c_in
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.weak3 = self._make_weak_learner(c_in, c_in_prev, c_out=self.in_planes,
                                             expansion=block.expansion,
                                             reduction=True, reduction_prev=True)
        # layer 4
        c_in, c_in_prev = self.in_planes, c_in
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.weak4 = self._make_weak_learner(c_in, c_in_prev, c_out=self.in_planes,
                                             expansion=block.expansion,
                                             reduction=True, reduction_prev=True)

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

    def _make_weak_learner(self, c_in, c_in_prev, c_out, expansion,
                           reduction, reduction_prev) -> nn.Module:
        if expansion > 1:
            return BottleneckBlock(
                genotype=self.genotype, num_layer=self.weak_layers,
                C_prev_prev=c_in_prev, C_prev=c_in, C=c_out // self.multiplier,
                expansion=expansion, reduction=reduction, reduction_prev=reduction_prev
            )
        else:
            return EvaluateBlock(
                genotype=self.genotype, num_layer=self.weak_layers,
                C_prev_prev=c_in_prev, C_prev=c_in, C=c_out // self.multiplier,
                reduction=reduction, reduction_prev=reduction_prev
            )

    def _forward_base_learner(self, x):
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

    def _forward_add_weak_learner(self, x):
        # --- stem ---
        out = self.conv1(x)
        out = self.bn1(out)
        # --- layers ---
        s_prev, s = out, out
        out = self.layer1(s) + self.weak1(s_prev, s, drop_prob=self.drop_prob)
        s_prev, s = s, out
        out = self.layer2(s) + self.weak2(s_prev, s, drop_prob=self.drop_prob)
        s_prev, s = s, out
        out = self.layer3(s) + self.weak3(s_prev, s, drop_prob=self.drop_prob)
        s_prev, s = s, out
        out = self.layer4(s) + self.weak4(s_prev, s, drop_prob=self.drop_prob)
        # --- global average pooling and classifier ---
        out = self.relu(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def stop_prop(self, is_stop=True):
        if is_stop:
            raise DeprecationWarning('set `stop_prop` to %s, but it is removed' % is_stop)
        else:
            logging.warning('set `stop_prop` to %s, but it is removed' % is_stop)

    def cal_flops(self):
        return sum([self.weak1.cal_flops(),
                    self.weak2.cal_flops() / 4,
                    self.weak3.cal_flops() / 16,
                    self.weak4.cal_flops() / 64])


def resnet18(genotype, **kwargs):
    return ResNet(genotype, ResNetBlock, [2, 2, 2, 2], **kwargs)

def resnet34(genotype, **kwargs):
    return ResNet(genotype, ResNetBlock, [3, 4, 6, 3], **kwargs)

def resnet50(genotype, **kwargs):
    return ResNet(genotype, ResNetBottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(genotype, **kwargs):
    return ResNet(genotype, ResNetBottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(genotype, **kwargs):
    return ResNet(genotype, ResNetBottleneck, [3, 8, 36, 3], **kwargs)
