import logging

import torch

import utils
from module.attacker import mfgsm
from module.validator import Validator


class MFGSMValidator(Validator):

    def __init__(self, model, criterion, valid_queue, n_repeats, decay, clip_eps,
                 data_mean, data_std):
        # valid basic
        self.model = model
        self.criterion = criterion
        self.valid_queue = valid_queue
        # adv training setting
        self.n_repeats = n_repeats
        self.decay = decay
        self.clip_eps = clip_eps
        # data statics
        self.data_mean = torch.tensor(data_mean).to('cuda').view(-1, 1, 1)
        self.data_std = torch.tensor(data_std).to('cuda').view(-1, 1, 1)

    def valid(self, report_freq=50):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.eval()

        logging.info('M-FGSM clip_eps: %.4f, n_repeats: %d'
                     % (self.clip_eps, self.n_repeats))

        for step, (image, target) in enumerate(self.valid_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            # add noise to batch
            image_pert = mfgsm(model=self.model, criterion=self.criterion,
                               image=image, target=target,
                               data_mean=self.data_mean, data_std=self.data_std,
                               clip_eps=self.clip_eps, decay=self.decay,
                               n_repeats=self.n_repeats)
            with torch.no_grad():
                image_pert.clamp_(0., 1.)  # clamp
                image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization
                # compute output
                output = self.model(image_pert)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)

                if step % report_freq == 0:
                    logging.info('m-fgsm valid %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                                 % (step, objs.avg, top1.avg, top5.avg))

        return top1.avg, objs.avg
