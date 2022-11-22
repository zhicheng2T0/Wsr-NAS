import logging

import torch

import utils
from module.searcher import Searcher


class DilationAdvSearcher(Searcher):

    def __init__(self, model, base_optimizer, boost_optimizer, arch_optimizer, criterion, train_queue, valid_queue,
                 n_repeats, fgsm_step, clip_eps, data_mean, data_std, batch_size, data_dim, crop_size,
                 flops_const, grad_clip=None):
        # training basic
        self.model = model
        self.base_optimizer = base_optimizer  # optimizer for base model
        self.boost_optimizer = boost_optimizer  # optimizer for boost model
        self.arch_optimizer = arch_optimizer  # optimizer for architecture parameter (\alpha)
        self.criterion = criterion
        self.train_queue = train_queue  # train set for network weights
        self.valid_queue = valid_queue  # valid set for architecture parameter
        self.grad_clip = grad_clip

        # loss params & optimization params
        self.alpha = 0.2  # for flops const
        self.beta = 0.6  # for flops const
        logging.info('alpha=%.4f, beta=%.4f' % (self.alpha, self.beta))

        # adv training setting
        self.n_repeats = n_repeats
        self.fgsm_step = fgsm_step
        self.clip_eps = clip_eps
        logging.info('n_repeats=%d, fgsm_step=%.4f, clip_eps=%.4f'
                     % (self.n_repeats, self.fgsm_step, self.clip_eps))

        # data statics
        self.data_mean = torch.tensor(data_mean).view(-1, 1, 1).to('cuda')
        self.data_std = torch.tensor(data_std).view(-1, 1, 1).to('cuda')

        # global perturbation
        self.global_pert = torch.zeros([batch_size, data_dim, crop_size, crop_size]).to('cuda')

        self.flops_const = flops_const
        print('flops_const: %s' % self.flops_const)

        self._zero_tenor_cuda = torch.tensor(0.).to('cuda')

    def base_step(self, optimizer, queue, report_freq):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.train()
        self.model.use_base()

        for step, (image, target) in enumerate(queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)
            image.sub_(self.data_mean).div_(self.data_std)  # normalization

            optimizer.zero_grad()
            self.model.zero_grad()

            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            # update meters
            prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('base std step %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (step, objs.avg, top1.avg, top5.avg))

        return top1.avg, objs.avg

    def boost_step(self, optimizer, queue, flops_const, report_freq):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.model.train()
        self.model.use_weak()

        for step, (image, target) in enumerate(queue):
            n = image.size(0)
            image = image.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)

            for n_batch_repeat in range(self.n_repeats):

                # --- adversarial forward ---

                # clear prev grad
                optimizer.zero_grad()
                self.model.zero_grad()

                # add noise to batch
                noise_batch = self.global_pert[:n].detach().clone().requires_grad_(True)  # get init from global noise
                image_pert = image + noise_batch  # add noise to batch
                image_pert.clamp_(0., 1.)  # clamp
                image_pert.sub_(self.data_mean).div_(self.data_std)  # normalization
                # adversarial loss of **f_adv**
                output = self.model(image_pert)
                adv_loss_of_f_adv = self.criterion(output, target)
                if flops_const:
                    flops = self.model.cal_flops() * 3 * 32 * 32
                    flops_factor = self.alpha * torch.log(flops) ** self.beta
                    if n_batch_repeat == (self.n_repeats - 1) and step % report_freq == 0:
                        print('flops %.2f, flops_factor %.4f' % (flops / 1e6, flops_factor))
                    (adv_loss_of_f_adv * flops_factor).backward()
                else:
                    adv_loss_of_f_adv.backward()

                # update **global perturbation**
                self.global_pert[:n] += self.fgsm_step * torch.sign(noise_batch.grad.detach())  # take a FGSM step
                self.global_pert.clamp_(-self.clip_eps, self.clip_eps)  # clip

                # update **weak learner**
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()  # update model

                if n_batch_repeat == (self.n_repeats - 1):
                    # --- update meters ---
                    prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                    objs.update(adv_loss_of_f_adv.item(), n)
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

            if step % report_freq == 0:
                logging.info('boost dilation-adv step %03d loss=%.2f top1-acc=%.2f top5-acc=%.2f'
                             % (step, objs.avg, top1.avg, top5.avg))

        return top1.avg, objs.avg

    def train(self, report_freq=50):
        # optimize base model
        for n_batch_repeat in range(1):
            base_top1, base_objs = self.base_step(
                optimizer=self.base_optimizer, queue=self.train_queue, report_freq=report_freq
            )
            logging.info('[base train overall] repeat-%02d train_acc=%.2f, train_loss=%.2f'
                         % (n_batch_repeat + 1, base_top1, base_objs))
        # optimize boost model
        # use `boost_optimizer`, not use `flops_const`, on `train_queue`
        boost_top1, boost_objs = self.boost_step(
            optimizer=self.boost_optimizer, queue=self.train_queue,
            flops_const=False, report_freq=report_freq
        )
        logging.info('[boost train overall] train_acc=%.2f, train_loss=%.2f' % (boost_top1, boost_objs))

    def search(self, report_freq=50):
        # optimize architecture of boost model (base model is frozen during search)
        # use `arch_optimizer` and `flops_const`, on `valid_queue`
        top1, objs = self.boost_step(
            optimizer=self.arch_optimizer, queue=self.valid_queue,
            flops_const=self.flops_const, report_freq=report_freq
        )
        logging.info('[boost search overall] train_acc=%.2f, train_loss=%.2f' % (top1, objs))
