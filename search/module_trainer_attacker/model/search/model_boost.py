import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from genotypes import PRIMITIVES, Genotype
from module.model.operations import OPS, FactorizedReduce, ReLUConvBN


def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.C = C
        self.mp = nn.MaxPool2d(2,2)
        self.k = 4
        for primitive in PRIMITIVES:
            op = OPS[primitive](C //self.k, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        #channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2//self.k, :, :]
        xtemp2 = x[ : ,  dim_2//self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
        else:
            if xtemp2.shape[-1] // 2 != temp1.shape[-1]:
                xtemp2 = F.pad(xtemp2, (0, 1, 0, 1), "constant", 0)
            # print(temp1.shape, xtemp2.shape)
            ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans,self.k)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans

    def cal_flops(self, weights):
        flops = 0.
        for w, op in zip(weights, self._ops):
            if isinstance(op, nn.Sequential):
                flops += w * 3 * 3 * (self.C // self.k)
            else:
                flops += w * op.flops_base
        return flops


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

    def cal_flops(self, weights, weights2):
        flops = 0.
        states = 2
        offset = 0
        for i in range(self._steps):
            for j in range(states):
                stride = 2 if self.reduction and j >= 2 else 1
                flops += weights2[offset + j] * self._ops[offset + j].cal_flops(weights[offset + j]) / stride / stride
            offset += states
            states += 1
        return flops


class SearchBlock(nn.Module):

    def __init__(self, num_layer, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(SearchBlock, self).__init__()
        self.num_layer = num_layer
        self.reduction = reduction
        layers = [Cell(steps=steps, multiplier=multiplier, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C,
                       reduction=reduction, reduction_prev=reduction_prev)]
        C_prev_prev, C_prev = C_prev, C * multiplier
        for _ in range(1, num_layer):
            layers.append(
                Cell(steps=steps, multiplier=multiplier, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C,
                     reduction=False, reduction_prev=reduction)
            )
            reduction = False
            C_prev_prev, C_prev = C_prev, C * multiplier
        self.layers = nn.ModuleList(layers)

    def forward(self, s0, s1, weights, weights2):

        out = None

        for layer in self.layers:
            # print(s0.size(), s1.size())
            out = layer(s0, s1, weights, weights2)
            s0, s1 = s1, out

        return out

    def cal_flops(self, weights, weights2):
        flops = 0.
        flops += self.layers[0].cal_flops(weights, weights2)
        stride = 2 if self.reduction else 1
        for layer in self.layers[1:]:
            flops += layer.cal_flops(weights, weights2) / stride / stride
        return flops


class BoostNetwork(nn.Module):

    def __init__(self, steps, multiplier):
        super(BoostNetwork, self).__init__()

        self._use_weak = False
        self.steps = steps
        self.multiplier = multiplier

        self._initialize_alphas()

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

    def _initialize_alphas(self):
        k = sum(1 for i in range(self.steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        # init architecture parameters alpha
        try:
            self.alphas = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
            self.betas = (1e-3 * torch.randn(k)).to('cuda').requires_grad_(True)
        except AssertionError:
            self.alphas = (1e-3 * torch.randn(k, num_ops)).requires_grad_(True)
            self.betas = (1e-3 * torch.randn(k)).requires_grad_(True)

    def cal_arch_weights(self):
        weights = torch.softmax(self.alphas, dim=-1)
        n = 3
        start = 2
        weights2 = torch.softmax(self.betas[0:2], dim=-1)
        for i in range(self.steps - 1):
            end = start + n
            tw2 = torch.softmax(self.betas[start:end], dim=-1)
            start = end
            n += 1
            weights2 = torch.cat([weights2, tw2], dim=0)
        return weights, weights2

    def arch_parameters(self):
        yield self.alphas
        yield self.betas

    def genotype(self):

        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')
                    )
                )[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        weights, weights2 = self.cal_arch_weights()

        gene = _parse(weights.data.to('cpu').numpy(), weights2.data.to('cpu').numpy())
        concat = range(2 + self.steps - self.multiplier, self.steps + 2)

        genotype = Genotype(ops=gene, ops_concat=concat)
        return genotype
