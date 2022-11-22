from typing import Union, List

import torch
from torch import nn, autograd
from module.operations import FactorizedReduce, OPS, ReLUConvBN
from genotypes import Genotype, PRIMITIVES
from utils import gumbel_like
from utils import gumbel_softmax_v1 as gumbel_softmax


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        idx = weights.argmax(dim=-1)
        return weights[idx] * self._ops[idx](x)
        # return sum(w * op(x) for w, op in zip(weights, self._ops))


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

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, tau, steps=4, multiplier=4, stem_multiplier=3):
        """
        :param C: init channels number
        :param num_classes: classes numbers
        :param layers: total number of layers
        :param criterion: loss function
        :param steps:
        :param multiplier:
        :param stem_multiplier:
        """
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._tau = tau

        # stem layer
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # body layers (normal and reduction)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to('cuda')
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = gumbel_softmax(self.alphas_reduce.data, tau=self._tau, dim=-1, g=self.g_reduce)
            else:
                weights = gumbel_softmax(self.alphas_normal.data, tau=self._tau, dim=-1, g=self.g_normal)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # calculate edge number k = (((1+1) + (steps+1)) * steps) / 2
        k = ((self._steps+3) * self._steps)//2
        # operations number
        num_ops = len(PRIMITIVES)

        # init architecture parameters alpha
        self.alphas_normal = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.g_normal = gumbel_like(self.alphas_normal)
        self.g_reduce = gumbel_like(self.alphas_reduce)

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas_normal, self.alphas_reduce]

    def arch_weights(self, cat: bool=True) -> Union[List[torch.tensor], torch.tensor]:
        weights = [
            gumbel_softmax(self.alphas_normal, tau=self._tau, dim=-1, g=self.g_normal),
            gumbel_softmax(self.alphas_reduce, tau=self._tau, dim=-1, g=self.g_reduce)
        ]
        if cat:
            return torch.cat(weights)
        else:
            return weights

    def genotype(self) -> Genotype:

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
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

        gene_normal = _parse(self.alphas_normal.detach().to('cpu').numpy())
        gene_reduce = _parse(self.alphas_reduce.detach().to('cpu').numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype



class Network_IN(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, tau, steps=4, multiplier=4, stem_multiplier=3):
        """
        :param C: init channels number
        :param num_classes: classes numbers
        :param layers: total number of layers
        :param criterion: loss function
        :param steps:
        :param multiplier:
        :param stem_multiplier:
        """
        super(Network_IN, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._tau = tau
        '''
        # stem layer
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # body layers (normal and reduction)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        '''


        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True#False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to('cuda')
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        #s0 = s1 = self.stem(input)

        s0=self.stem0(input)
        s1=self.stem1(s0)
        #print('input',input.shape,'s0',s0.shape,'s1',s1.shape)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = gumbel_softmax(self.alphas_reduce.data, tau=self._tau, dim=-1, g=self.g_reduce)
            else:
                weights = gumbel_softmax(self.alphas_normal.data, tau=self._tau, dim=-1, g=self.g_normal)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # calculate edge number k = (((1+1) + (steps+1)) * steps) / 2
        k = ((self._steps+3) * self._steps)//2
        # operations number
        num_ops = len(PRIMITIVES)

        # init architecture parameters alpha
        self.alphas_normal = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.g_normal = gumbel_like(self.alphas_normal)
        self.g_reduce = gumbel_like(self.alphas_reduce)

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas_normal, self.alphas_reduce]

    def arch_weights(self, cat: bool=True) -> Union[List[torch.tensor], torch.tensor]:
        weights = [
            gumbel_softmax(self.alphas_normal, tau=self._tau, dim=-1, g=self.g_normal),
            gumbel_softmax(self.alphas_reduce, tau=self._tau, dim=-1, g=self.g_reduce)
        ]
        if cat:
            return torch.cat(weights)
        else:
            return weights

    def genotype(self) -> Genotype:

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
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

        gene_normal = _parse(self.alphas_normal.detach().to('cpu').numpy())
        gene_reduce = _parse(self.alphas_reduce.detach().to('cpu').numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
