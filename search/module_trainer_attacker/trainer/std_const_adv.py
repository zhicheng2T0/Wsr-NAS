import logging

import torch
from torch.nn import DataParallel

import utils
from module.attacker import pgd_attack
from module.trainer import Trainer


class StdConstAdvTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, train_queue,
                 n_repeats, fgsm_step, clip_eps, data_mean, data_std,
                 grad_clip=5.0):
        # training basic
        self.model = model
        self.optimizer = optimizer  # optimizer for boost model
        self.criterion = criterion
        self.train_queue = train_queue  # train set for network weights

        # loss params & optimization params
        self.lamb_a = 0.0  # for lagrangian
        self.rho = 0.001  # for lagrangian
        logging.info('rho=%.4f' % (self.rho,))

        # adv training setting
        self.n_repeats = n_repeats
        self.fgsm_step = fgsm_step
        self.clip_eps = clip_eps
        logging.info('n_repeats=%d, fgsm_step=%.4f, clip_eps=%.4f'
                     % (self.n_repeats, self.fgsm_step, self.clip_eps))

        # data statics
        self.data_mean = torch.tensor(data_mean).view(-1, 1, 1).to('cuda')
        self.data_std = torch.tensor(data_std).view(-1, 1, 1).to('cuda')

        self.grad_clip = grad_clip

        self._zero_tenor_cuda = torch.tensor(0.).to('cuda')

    def _step(self, optimizer, lamb, queue, report_freq):
        objs = utils.AverageMeter()
        objs_adv = utils.AverageMeter()
        objs_const = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.train()

        for step, (image, target) in enumerate(queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            # --- standard forward ---

            # clear prev grad
            optimizer.zero_grad()
            self.model.zero_grad()

            image_norm = image.detach().clone().sub_(self.data_mean).div_(self.data_std)  # normalization
            # standard loss of **f_std**
            if isinstance(self.model, DataParallel):
                self.model.module.use_base()
            else:
                self.model.use_base()

            with torch.no_grad():
                output = self.model(image_norm)
                std_loss_of_f_std = self.criterion(output, target)
            # standard loss of **f_adv**
            if isinstance(self.model, DataParallel):
                self.model.module.use_weak()
            else:
                self.model.use_weak()

            output = self.model(image_norm)
            std_loss_of_f_adv = self.criterion(output, target)
            # the constraint term (c <= 0)
            std_perf_constraint = std_loss_of_f_adv - std_loss_of_f_std

            # TODO: balance the weight of the constraint
            loss_const = lamb * std_perf_constraint \
                         + self.rho / 2 * torch.max(std_perf_constraint, self._zero_tenor_cuda) ** 2
            loss_const.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # update model
            optimizer.step()

            # --- adversarial forward ---

            # clear prev grad
            optimizer.zero_grad()
            self.model.zero_grad()
            if isinstance(self.model, DataParallel):
                self.model.module.use_weak()
            else:
                self.model.use_weak()

            # generate adversarial sample
            all_pert = pgd_attack(model=self.model, criterion=self.criterion,
                                  image=image, target=target,
                                  data_mean=self.data_mean, data_std=self.data_std,
                                  clip_eps=self.clip_eps, fgsm_step=self.fgsm_step,
                                  n_repeats=self.n_repeats, pert_at=None)

            # add noise to batch
            assert len(all_pert) == 1
            image_pert = image + all_pert[self.n_repeats]  # add noise to batch
            image_pert.clamp_(0., 1.)  # clamp
            image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization
            image_pert = image_pert.requires_grad_(False)
            # adversarial loss of **f_adv**
            output = self.model(image_pert)
            adv_loss_of_f_adv = self.criterion(output, target)
            adv_loss_of_f_adv.backward()

            # update **weak learner**
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()  # update model

            # --- update meters ---
            prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
            objs.update((adv_loss_of_f_adv + loss_const).item(), n)
            objs_adv.update(adv_loss_of_f_adv.item(), n)
            objs_const.update(std_perf_constraint.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)
            # --- update lambda ---
            # lamb += self.rho * std_perf_constraint.detach().clone()

            if step % report_freq == 0:
                logging.info('boost std-const-adv step %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (step, objs.avg, top1.avg, top5.avg))
                logging.info(' - loss-adv=%.2f std-const=%.2f lambda=%.3e'
                             % (objs_adv.avg, objs_const.avg, lamb))

        lamb += self.rho * objs_const.avg
        return top1.avg, objs.avg, lamb

    def train(self, report_freq=50):
        # optimize architecture of boost model (base model is frozen during search)
        # use `arch_optimizer` and `flops_const`, on `valid_queue`
        top1, objs, self.lamb_a = self._step(
            optimizer=self.optimizer, lamb=self.lamb_a, queue=self.train_queue,
            report_freq=report_freq
        )
        logging.info('[boost train overall] train_acc=%.2f, train_loss=%.2f' % (top1, objs))
        return top1, objs
