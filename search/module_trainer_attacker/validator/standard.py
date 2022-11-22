import logging

import torch

import utils
from module.validator import Validator


class StandardValidator(Validator):

    def __init__(self, model, criterion, valid_queue, data_mean, data_std):
        # valid basic
        self.model = model
        self.criterion = criterion
        self.valid_queue = valid_queue
        # data statics
        self.data_mean = torch.tensor(data_mean).to('cuda').view(-1, 1, 1)
        self.data_std = torch.tensor(data_std).to('cuda').view(-1, 1, 1)

    @torch.no_grad()
    def valid(self, report_freq=50):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.eval()

        for step, (image, target) in enumerate(self.valid_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            # compute output
            image.sub_(self.data_mean).div_(self.data_std) # normalization
            output = self.model(image)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('std valid %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (step, objs.avg, top1.avg, top5.avg))

        return top1.avg, objs.avg
