import logging

import torch

import utils
from module.attacker import fgsm_attack
from module.validator import Validator


class BlackBoxFGSMValidator(Validator):

    def __init__(self, source_model, target_model, criterion, valid_queue, fgsm_step, data_mean, data_std):
        # valid basic
        self.source_model = source_model
        self.target_model = target_model
        self.criterion = criterion
        self.valid_queue = valid_queue
        self.fgsm_step = fgsm_step
        # data statics
        self.data_mean = torch.tensor(data_mean).to('cuda').view(-1, 1, 1)
        self.data_std = torch.tensor(data_std).to('cuda').view(-1, 1, 1)

    def valid(self, report_freq=50):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.source_model.eval()
        self.target_model.eval()

        for step, (image, target) in enumerate(self.valid_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            # add noise to batch
            image_pert = fgsm_attack(model=self.source_model, criterion=self.criterion,
                                     image=image, target=target,
                                     data_mean=self.data_mean, data_std=self.data_std,
                                     fgsm_step=self.fgsm_step)
            with torch.no_grad():
                image_pert.clamp_(0., 1.)  # clamp
                image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization
                # compute output
                output = self.target_model(image_pert)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)

                if step % report_freq == 0:
                    logging.info('fgsm valid %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                                 % (step, objs.avg, top1.avg, top5.avg))

        return top1.avg, objs.avg
