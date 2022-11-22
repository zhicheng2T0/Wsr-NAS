from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from module.model.evaluate import BoostNetwork, EvaluateBlock


class SmallCNN(BoostNetwork):
    def __init__(self, genotype, data_dim=3, num_classes=10, drop_prob=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = data_dim
        self.num_labels = num_classes

        self.genotype = genotype
        self.multiplier = len(self.genotype.ops_concat)
        self.drop_prob = drop_prob

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.weak1 = EvaluateBlock(genotype=self.genotype, num_layer=1,
                                   C_prev_prev=self.num_channels, C_prev=self.num_channels, C=32//self.multiplier,
                                   reduction=False, reduction_prev=False)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.weak2 = EvaluateBlock(genotype=self.genotype, num_layer=1,
                                   C_prev_prev=self.num_channels, C_prev=32, C=32//self.multiplier,
                                   reduction=False, reduction_prev=True)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.weak3 = EvaluateBlock(genotype=self.genotype, num_layer=1,
                                   C_prev_prev=32, C_prev=32, C=64 // self.multiplier,
                                   reduction=False, reduction_prev=True)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop_prob)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def _forward_base_learner(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

    def _forward_add_weak_learner(self, x):

        features = self.feature_extractor(x)

        out = self.weak1(x, x, drop_prob=self.drop_prob)
        out1 = self.maxpool1(out)

        out = self.weak2(x, out1, drop_prob=self.drop_prob)
        out2 = self.maxpool2(out)

        out = self.weak3(out1, out2, drop_prob=self.drop_prob)
        out = F.pad(out, (0, 1, 0, 1), "constant", 0)
        out3 = self.maxpool3(out)

        features += out3

        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

    def cal_flops(self):
        return sum([self.weak1.cal_flops(),
                    self.weak2.cal_flops() / 4])
