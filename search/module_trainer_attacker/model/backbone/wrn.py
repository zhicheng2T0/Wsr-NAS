import torch
import torch.nn as nn

import torch.nn.functional as F


class WRNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(WRNBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.drop_rate = drop_rate

        self.relu = nn.ReLU(inplace=True)

        # --- ReLU-Conv-BN-1 ---
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # --- ReLU-Conv-BN-2 ---
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # --- residual conn ---
        if self.in_planes == self.out_planes:
            self.conv_shortcut = None
        else:
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):

        # --- ReLU-Conv-BN-1 ---
        identity = out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        # --- ReLU-Conv-BN-2 ---
        out = self.relu(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = self.bn2(out)

        # --- residual conn ---
        if self.conv_shortcut is not None:
            identity = self.conv_shortcut(identity)
        out += identity
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=32, data_dim=3, num_classes=10, widen_factor=10, drop_prob=0.):
        super(WideResNet, self).__init__()

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 2) % 6 == 0)
        n = (depth - 2) // 6
        block = WRNBlock

        self.drop_prob = drop_prob

        # --- stem ---
        self.conv1 = nn.Conv2d(data_dim, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])

        # --- layers ---
        # layer 1
        self.layer1 = self._make_layer(block, n_channels[0], n_channels[1], n, stride=1, drop_rate=drop_prob)
        # layer 2
        self.layer2 = self._make_layer(block, n_channels[1], n_channels[2], n, stride=2, drop_rate=drop_prob)
        # layer 3
        self.layer3 = self._make_layer(block, n_channels[2], n_channels[3], n, stride=2, drop_rate=drop_prob)

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

    def forward(self, x):
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
