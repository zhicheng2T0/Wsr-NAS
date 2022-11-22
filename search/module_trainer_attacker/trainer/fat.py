import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import utils
from module.trainer import Trainer


def early_stop_attack(model, data, target, data_mean, data_std,
                      step_size, epsilon, perturb_steps, tau,
                      random_init_type, loss_fn, rand_init=True, omega=0.):

    model.eval()

    K = perturb_steps
    count = 0
    output_target = []
    output_adv = []
    output_natural = []

    control = (torch.ones(len(target)) * tau).cuda()

    # Initialize the adversarial data with random noise
    if rand_init:
        if random_init_type == "normal_distribution_random_init":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        elif random_init_type == "uniform_random_init":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        else:
            raise ValueError('unknown random_init_type: %s' % random_init_type)
    else:
        iter_adv = data.cuda().detach()

    iter_clean_data = data.cuda().detach()
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data.sub(data_mean).div(data_std))

    while K > 0:
        iter_adv.requires_grad_()
        output = model(iter_adv.sub(data_mean).div(data_std))
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)

        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_target) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)

        # calculate gradient
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
            count += len(iter_target)
        else:
            output_adv = output_adv.detach()
            return output_adv, output_target, output_natural, count
        K = K-1

    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
    output_adv = output_adv.detach()
    return output_adv, output_target, output_natural, count


def adjust_tau(tau, epoch, dynamic_tau):
    if dynamic_tau:
        if epoch <= 50:
            tau = 0
        elif epoch <= 90:
            tau = 1
        else:
            tau = 2
    return tau


class FATTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, train_queue, n_repeats, fgsm_step, clip_eps,
                 data_mean, data_std, tau=0, dynamic_tau=True, rand_init=True, omega=0.001, grad_clip=5.0):
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

        self.grad_clip = grad_clip

        self.tau = tau
        self.dynamic_tau = dynamic_tau
        self.rand_init = rand_init
        self.omega = omega

        self._epoch_counter = 0

    def train(self, report_freq=50):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        bp_count = 0

        for step, (image, target) in enumerate(self.train_queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            # get friendly adversarial training data via early-stopped PGD
            output_adv, output_target, output_natural, count = early_stop_attack(
                model=self.model,
                data=image, target=target,
                data_mean=self.data_mean, data_std=self.data_std,
                step_size=self.fgsm_step,
                epsilon=self.clip_eps,
                perturb_steps=self.n_repeats,
                tau=adjust_tau(tau=self.tau, epoch=self._epoch_counter, dynamic_tau=self.dynamic_tau),
                random_init_type="uniform_random_init",
                loss_fn='cent',
                rand_init=self.rand_init,
                omega=self.omega
            )

            bp_count += count

            # update model with friendly AT data
            self.model.train()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            output = self.model(output_adv.sub(self.data_mean).div(self.data_std))
            loss = self.criterion(output, output_target)
            loss.backward()
            self.optimizer.step()

            # update meters
            prec1, prec5 = utils.accuracy(output, output_target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('fat-%d train %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (self.n_repeats, step, objs.avg, top1.avg, top5.avg))

        bp_count_avg = bp_count / len(self.train_queue.dataset)

        self._epoch_counter += 1
        logging.info('[train overall] train_acc=%.2f, train_loss=%.2f, bp_count_avg=%.2f', top1.avg, objs.avg, bp_count_avg)
        return top1.avg, objs.avg
