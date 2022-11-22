import logging

import torch

import utils
from module.attacker import pgd_attack
from module.validator import Validator


class PGDValidator(Validator):

    def __init__(self, model, criterion, valid_queue, n_repeats, fgsm_step, clip_eps,
                 data_mean, data_std, eval_at=None):
        # valid basic
        self.model = model
        self.criterion = criterion
        self.valid_queue = valid_queue
        # adv training setting
        self.n_repeats = n_repeats
        self.fgsm_step = fgsm_step
        self.clip_eps = clip_eps
        # data statics
        self.data_mean = torch.tensor(data_mean).to('cuda').view(-1, 1, 1)
        self.data_std = torch.tensor(data_std).to('cuda').view(-1, 1, 1)
        # eval_at
        if eval_at is None:
            self.eval_at = [self.n_repeats]
        else:
            self.eval_at = eval_at
            if self.n_repeats not in self.eval_at:
                self.eval_at.append(self.n_repeats)

    def valid(self, report_freq=50):
        objs = {i: utils.AverageMeter() for i in self.eval_at}
        top1 = {i: utils.AverageMeter() for i in self.eval_at}
        top5 = {i: utils.AverageMeter() for i in self.eval_at}
        self.model.eval()

        logging.info('PGD clip_eps: %.4f, n_repeats: %d, fgsm_step: %.4f '
                     % (self.clip_eps, self.n_repeats, self.fgsm_step))

        for step, (image, target) in enumerate(self.valid_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            all_pert = pgd_attack(model=self.model, criterion=self.criterion,
                                  image=image, target=target,
                                  data_mean=self.data_mean, data_std=self.data_std,
                                  clip_eps=self.clip_eps, fgsm_step=self.fgsm_step,
                                  n_repeats=self.n_repeats, pert_at=self.eval_at)

            assert len(all_pert) == len(self.eval_at)

            # evaluate
            with torch.no_grad():
                for k in all_pert:
                    image_pert = image + all_pert[k]  # add noise to batch
                    image_pert.clamp_(0., 1.)  # clamp
                    image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization
                    # compute output
                    output = self.model(image_pert)
                    loss = self.criterion(output, target)

                    # measure accuracy and record loss
                    prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                    objs[k].update(loss.item(), n)
                    top1[k].update(prec1.data.item(), n)
                    top5[k].update(prec5.data.item(), n)

                    if step % report_freq == 0:
                        logging.info('pgd-%02d valid %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                                     % (k, step, objs[k].avg, top1[k].avg, top5[k].avg))

        top1_avg = {k: top1[k].avg for k in top1}
        objs_avg = {k: objs[k].avg for k in objs}
        return top1_avg, objs_avg
