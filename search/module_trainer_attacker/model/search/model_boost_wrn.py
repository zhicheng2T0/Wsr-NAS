import torch
import torch.nn as nn

from module.model.backbone.wrn import WRNBlock
from module.model.search import BoostNetwork, SearchBlock


class WideResNet(BoostNetwork):
    def __init__(self, depth=32, num_classes=10, widen_factor=10, data_dim=3,
                 steps=4, multiplier=4, drop_rate=0.):
        super(WideResNet, self).__init__(steps, multiplier)

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 2) % 6 == 0)
        n = (depth - 2) // 6
        block = WRNBlock

        # --- stem ---
        self.conv1 = nn.Conv2d(data_dim, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])

        # --- layers ---
        # layer 1
        c_in = c_in_prev = n_channels[0]
        self.layer1 = self._make_layer(block, n_channels[0], n_channels[1], n, stride=1, drop_rate=drop_rate)
        self.weak1 = self._make_weak_learner(c_in, c_in_prev, c_out=n_channels[1],
                                             reduction=False, reduction_prev=False)
        # layer 2
        c_in, c_in_prev = n_channels[1], c_in
        self.layer2 = self._make_layer(block, n_channels[1], n_channels[2], n, stride=2, drop_rate=drop_rate)
        self.weak2 = self._make_weak_learner(c_in, c_in_prev, c_out=n_channels[2],
                                             reduction=True, reduction_prev=False)
        # layer 3
        c_in, c_in_prev = n_channels[2], c_in
        self.layer3 = self._make_layer(block, n_channels[2], n_channels[3], n, stride=2, drop_rate=drop_rate)
        self.weak3 = self._make_weak_learner(c_in, c_in_prev, c_out=n_channels[3],
                                             reduction=True, reduction_prev=True)

        # --- global average pooling and classifier ---
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = [block(in_planes, out_planes, stride=stride, drop_rate=drop_rate)]
        for i in range(1, nb_layers):
            layers.append(block(out_planes, out_planes, stride=1, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def _make_weak_learner(self, c_in, c_in_prev, c_out, reduction, reduction_prev) -> nn.Module:
        # print(c_in, c_in_prev, c_out)
        return SearchBlock(num_layer=3, steps=self.steps, multiplier=self.multiplier,
                           C_prev_prev=c_in_prev, C_prev=c_in, C=c_out//self.multiplier,
                           reduction=reduction, reduction_prev=reduction_prev)

    def _forward_base_learner(self, x):
        # --- stem ---
        out = self.conv1(x)
        out = self.bn1(out)
        # --- layers ---
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # --- global average pooling and classifier ---
        out = self.relu(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def _forward_add_weak_learner(self, x):
        # cal weights
        weights, weights2 = self.cal_arch_weights()

        # --- stem ---
        out = self.conv1(x)
        out = self.bn1(out)
        # --- layers ---
        s_prev, s = out, out
        out = self.layer1(s) + self.weak1(s_prev, s, weights, weights2)
        s_prev, s = s, out
        out = self.layer2(s) + self.weak2(s_prev, s, weights, weights2)
        s_prev, s = s, out
        out = self.layer3(s) + self.weak3(s_prev, s, weights, weights2)
        # --- global average pooling and classifier ---
        out = self.relu(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def cal_flops(self):
        weights, weights2 = self.cal_arch_weights()
        weights = weights.detach()
        weights2 = weights2.detach()
        return sum([self.weak1.cal_flops(weights, weights2),
                    self.weak2.cal_flops(weights, weights2) / 4,
                    self.weak3.cal_flops(weights, weights2) / 16])
