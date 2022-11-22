import torch
import torch.nn as nn
from torch import Tensor

from module.model.operations import OPS, FactorizedReduce, ReLUConvBN, Identity
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.C = C
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # --- unpack genotype ---
        self._op_names, self._indices = zip(*genotype.ops)
        self._concat = genotype.ops_concat

        # --- compile model ---
        assert len(self._op_names) == len(self._indices)
        self._steps = len(self._op_names) // 2
        self.multiplier = len(self._concat)
        self._ops = nn.ModuleList()
        for name, index in zip(self._op_names, self._indices):
            stride = 2 if self.reduction and index < 2 else 1
            op = OPS[name](self.C, stride, True)
            self._ops += [op]

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

    def cal_flops(self):
        flops = 0.
        for op, index in zip(self._ops, self._indices):
            stride = 2 if self.reduction and index >= 2 else 1
            if isinstance(op, (nn.AvgPool2d, nn.MaxPool2d)):
                flops += 3 * 3 * self.C / stride / stride
            else:
                flops += op.flops_base / stride / stride
        return flops


class EvaluateBlock(nn.Module):

    def __init__(self, genotype, num_layer, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(EvaluateBlock, self).__init__()
        self.num_layer = num_layer
        self.reduction = reduction
        layers = [Cell(genotype=genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C,
                       reduction=reduction, reduction_prev=reduction_prev)]
        C_prev_prev, C_prev = C_prev, C * layers[-1].multiplier
        for _ in range(1, num_layer):
            layers.append(
                Cell(genotype=genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C,
                     reduction=False, reduction_prev=reduction)
            )
            reduction = False
            C_prev_prev, C_prev = C_prev, C * layers[-1].multiplier
        self.layers = nn.ModuleList(layers)

    def forward(self, s0, s1, drop_prob):

        out = None

        for layer in self.layers:
            # print(s0.size(), s1.size())
            out = layer(s0, s1, drop_prob=drop_prob)
            s0, s1 = s1, out

        return out

    def cal_flops(self):
        flops = 0.
        flops += self.layers[0].cal_flops()
        stride = 2 if self.reduction else 1
        for layer in self.layers[1:]:
            flops += layer.cal_flops() / stride / stride
        return flops


class BoostNetwork(nn.Module):

    def __init__(self):
        super(BoostNetwork, self).__init__()

        self._use_weak = False

    def _forward_base_learner(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def _forward_add_weak_learner(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def cal_flops(self):
        raise NotImplementedError()

    def use_base(self):
        self._use_weak = False

    def use_weak(self):
        self._use_weak = True

    def forward(self, x: Tensor) -> Tensor:
        if self._use_weak:
            return self._forward_add_weak_learner(x)
        else:
            return self._forward_base_learner(x)

    def base_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'weak' != name[:4]:
                # print(name)
                yield param

    def weak_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'weak' == name[:4]:
                # print(name)
                yield param
