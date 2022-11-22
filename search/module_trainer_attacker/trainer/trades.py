import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import utils
from module.trainer import Trainer


def trades_loss(model,
                x_natural,
                y,
                data_mean,
                data_std,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to('cuda')
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv.sub(data_mean).div(data_std)), dim=1),
                                       F.softmax(model(x_natural.sub(data_mean).div(data_std)), dim=1))
            grad = torch.autograd.grad(loss_kl, x_adv, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to('cuda').detach()
        delta = delta.requires_grad_(True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv.sub(data_mean).div(data_std)), dim=1),
                                           F.softmax(model(x_natural.sub(data_mean).div(data_std)), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = (x_natural + delta).detach().requires_grad_(False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = x_adv.clamp(0., 1.).detach().requires_grad_(False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits_nat = model(x_natural.sub(data_mean).div(data_std))
    logits_adv = model(x_adv.sub(data_mean).div(data_std))
    loss_natural = F.cross_entropy(logits_nat, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_nat, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, logits_adv


class TradesTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, train_queue,
                 n_repeats, fgsm_step, clip_eps, data_mean, data_std, beta=6.0,
                 grad_clip=5.0):
        # training basic
        self.model = model
        self.optimizer = optimizer  # optimizer for boost model
        self.criterion = criterion
        self.train_queue = train_queue  # train set for network weights

        # adv training setting
        self.n_repeats = n_repeats
        self.fgsm_step = fgsm_step
        self.clip_eps = clip_eps
        self.beta = beta
        logging.info('n_repeats=%d, fgsm_step=%.4f, clip_eps=%.4f'
                     % (self.n_repeats, self.fgsm_step, self.clip_eps))

        # data statics
        self.data_mean = torch.tensor(data_mean).view(-1, 1, 1).to('cuda')
        self.data_std = torch.tensor(data_std).view(-1, 1, 1).to('cuda')

        self.grad_clip = grad_clip

    def train(self, report_freq=50):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.train()

        for step, (image, target) in enumerate(self.train_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)
            self.optimizer.zero_grad()

            # calculate robust loss
            loss, output = trades_loss(model=self.model,
                                       x_natural=image,
                                       y=target,
                                       data_mean=self.data_mean,
                                       data_std=self.data_std,
                                       optimizer=self.optimizer,
                                       step_size=self.fgsm_step,
                                       epsilon=self.clip_eps,
                                       perturb_steps=self.n_repeats,
                                       beta=self.beta)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # update meters
            prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('trades-%d train %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (self.n_repeats, step, objs.avg, top1.avg, top5.avg))

        logging.info('[train overall] train_acc=%.2f, train_loss=%.2f', top1.avg, objs.avg)
        return top1.avg, objs.avg
