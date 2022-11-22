import logging

import torch
from torch import nn
from torch.nn import functional as F

import utils
from module.attacker import pgd_attack
from module.trainer import Trainer


class PGDTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, train_queue, n_repeats, fgsm_step, clip_eps,
                 data_mean, data_std, teacher=None, T=1, grad_clip=5.0):
        # training basic
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_queue = train_queue
        # adv training setting
        self.n_repeats = n_repeats
        self.fgsm_step = fgsm_step
        self.clip_eps = clip_eps
        # data statics
        self.data_mean = torch.tensor(data_mean).view(-1, 1, 1).to('cuda')
        self.data_std = torch.tensor(data_std).view(-1, 1, 1).to('cuda')
        self.teacher = teacher
        self.T = T

        self.grad_clip = grad_clip

    def train(self, report_freq=50, constraint=None):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.train()

        for step, (image, target) in enumerate(self.train_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            # generate adversarial sample
            all_pert = pgd_attack(model=self.model, criterion=self.criterion,
                                  image=image, target=target,
                                  data_mean=self.data_mean, data_std=self.data_std,
                                  clip_eps=self.clip_eps, fgsm_step=self.fgsm_step,
                                  n_repeats=self.n_repeats, pert_at=None)
            assert len(all_pert) == 1
            image_pert = image + all_pert[self.n_repeats]  # add noise to batch
            image_pert.clamp_(0., 1.)  # clamp
            image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization

            # clear prev grad
            self.optimizer.zero_grad()
            self.model.zero_grad()

            # compute current grad
            output = self.model(image_pert)
            loss = self.criterion(output, target)
            if self.teacher is not None:
                teacher_logits = self.teacher(image)[1]
                loss = F.kl_div(F.log_softmax(output / self.T), F.softmax(teacher_logits / self.T)) * self.T * self.T
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # update model
            self.optimizer.step()

            if constraint is not None:
                constraint(image, target)

            # update meters
            prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('pgd-%d train %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (self.n_repeats, step, objs.avg, top1.avg, top5.avg))

        logging.info('[train overall] train_acc=%.2f, train_loss=%.2f', top1.avg, objs.avg)
        return top1.avg, objs.avg
