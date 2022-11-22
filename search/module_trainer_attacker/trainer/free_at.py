import logging

import torch
from torch.nn import functional as F

import utils
from module_trainer_attacker.trainer import Trainer


class FreeATTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, train_queue, n_repeats, fgsm_step, clip_eps,
                 data_mean, data_std, batch_size, crop_size, teacher=None, T=1, grad_clip=5.0, break_value=-1):
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
        # global perturbation
        self.global_pert = torch.zeros([batch_size, 3, crop_size, crop_size]).to('cuda')

        self.teacher = teacher
        self.T = T
        self.break_value=break_value

        self.grad_clip = grad_clip

    def train(self, report_freq=200,case=False):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.train()

        for step, (image, target) in enumerate(self.train_queue):
            if step==self.break_value:
                break
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            for n_batch_repeat in range(self.n_repeats):
                # clear prev grad
                self.optimizer.zero_grad()
                self.model.zero_grad()

                # add noise to batch
                noise_batch = self.global_pert[:n].detach().clone().requires_grad_(True)  # get init from global noise
                #print(image.shape,noise_batch.shape)
                image_pert = image + noise_batch  # add noise to batch
                image_pert.clamp_(0., 1.)  # clamp
                image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization
                # compute current grad
                if case==False:
                    output,_ = self.model(image_pert)
                else:
                    output = self.model(image_pert)
                loss = self.criterion(output, target)
                if self.teacher is not None:
                  teacher_logits = self.teacher(image)[1]
                  loss = F.kl_div(F.log_softmax(output/self.T), F.softmax(teacher_logits/self.T)) * self.T * self.T
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # update global perturbation
                self.global_pert[:n] += self.fgsm_step * torch.sign(noise_batch.grad.detach())  # take a FGSM step
                self.global_pert.clamp_(-self.clip_eps, self.clip_eps)  # clip
                # update model
                self.optimizer.step()
                # update meters
                if n_batch_repeat == (self.n_repeats - 1):
                    prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                    objs.update(loss.item(), n)
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('free-at train %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (step, objs.avg, top1.avg, top5.avg))

        logging.info('[train overall] train_acc=%.2f, train_loss=%.2f', top1.avg, objs.avg)
        return top1.avg, objs.avg
