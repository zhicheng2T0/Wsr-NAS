import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from trades import trades_loss, madry_loss

from utils import pgd_attack
from torch.nn.parallel import DistributedDataParallel
from util.dist_init import dist_init
from util.torch_dist_sum import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data_folder/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size') #128
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=201, help='num of training epochs')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation')
parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
parser.add_argument('--step_size', type=float, default=0.01, help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0, help='regularization in TRADES')
parser.add_argument('--adv_loss', type=str, default='pgd', help='experiment name')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='ADVRUSH', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--port', type=int, default=23456, help='master port')
parser.add_argument('--break_value', type=int, default=-1, help='break value in train val loops')
parser.add_argument('--break_epoch', type=int, default=176, help='break epoch for early stopping')
parser.add_argument('--perturb_steps', type=int, default=10, help='steps to be perturbed in TRADES')

args = parser.parse_args()

rank, local_rank, world_size = dist_init(port=args.port)


args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if rank==0:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)
if rank==0:
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)



CIFAR_CLASSES = 10


def main():
  actual_bs=args.batch_size // world_size
  extra_strengths=[0.17,0.25]
  full_ae_strength_list=[0.03,0.045,0.06,0.075,0.09,0.105]
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  

  np.random.seed(args.seed)
  torch.cuda.set_device(rank)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  if rank==0:
    logging.info('gpu device = %d' % rank)
    logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  model.drop_path_prob=0

  model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False)
  if rank==0:
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10_eval(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  test_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

  train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=actual_bs, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, sampler=train_sampler)

  valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=actual_bs//2, shuffle=(test_sampler is None),
        num_workers=2, pin_memory=True, sampler=test_sampler)

  best_acc = 0.0
  for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    if rank==0:
      logging.info('epoch %d train_acc %f', epoch, train_acc)
    
    infer_clean(valid_queue, model, criterion)

    if (epoch==5 or epoch%25==0) and epoch!=0:
        if epoch>args.break_epoch:
          break
        valid_acc, valid_obj = infer(valid_queue, model, criterion,full_ae_strength_list,extra_strengths,20)
        if valid_acc > best_acc:
          best_acc = valid_acc
          utils.save_checkpoint({
            'epoch': epoch +1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, is_best=True, save=args.save, epoch=epoch)
        if rank==0:
          logging.info('epoch %d valid_acc %f, best_acc %f', epoch, valid_acc, best_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        utils.save_checkpoint({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, is_best=False, save=args.save, epoch=epoch)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  break_step=args.break_value

  for step, (input, target) in enumerate(train_queue):
    if step==break_step:
        break
    input = Variable(input).cuda(non_blocking=True)
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    if args.adv_loss == 'pgd':
      loss = madry_loss(
            model,
            input, 
            target, 
            optimizer,
            step_size = args.epsilon*2.5/args.num_steps,
            epsilon = args.epsilon, 
            perturb_steps = args.num_steps)
    elif args.adv_loss == 'trades':
      loss = trades_loss(model,
                input,
                target,
                optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                distance='l_inf')
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0 and rank==0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion,full_ae_strength_list,extra_strengths,ae_steps):
    objs_clean = utils.AvgrageMeter()
    top1_clean = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    break_step=args.break_value

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if step==break_step:
                break
        
            input = Variable(input, requires_grad=False).cuda(non_blocking=True)
            target = Variable(target, requires_grad=False).cuda(non_blocking=True)

            logits,_ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs_clean.update(loss.data.item(), n)
            top1_clean.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    sum1, cnt1, sumobj, cntobj = torch_dist_sum(local_rank, top1_clean.sum, top1_clean.cnt, objs_clean.sum, objs_clean.cnt)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    obj__ = sum(sumobj.float()) / sum(cntobj.float())
    if rank==0:
      logging.info('valid %f %f', obj__, top1_acc)


    for i in range(len(full_ae_strength_list)):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(valid_queue):
            if step==break_step:
                break
            # data to CUDA
            input = input.to('cuda')
            target = target.to('cuda')
            
            model.train()
            input,_ = pgd_attack(model, input, target, criterion,eps=full_ae_strength_list[i],alpha=full_ae_strength_list[i]*(2.5/ae_steps),iters=ae_steps)
            model.eval()
            logits,_ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)
        sum1, cnt1, sumobj, cntobj = torch_dist_sum(local_rank, top1.sum, top1.cnt, objs.sum, objs.cnt)
        top1_acc = sum(sum1.float()) / sum(cnt1.float())
        obj__ = sum(sumobj.float()) / sum(cntobj.float())
        if rank==0:
          logging.info('valid (ae) eps=%.4e loss=%.4e top1-acc=%.4f ',full_ae_strength_list[i], obj__, top1_acc)

    if extra_strengths!=None:
        for i in range(len(extra_strengths)):
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()
            model.train()

            for step, (input, target) in enumerate(valid_queue):
                if step==break_step:
                    break
                # data to CUDA
                input = input.to('cuda')
                target = target.to('cuda')
                model.train()
                input,_ = pgd_attack(model, input, target, criterion,eps=extra_strengths[i],alpha=extra_strengths[i]*(2.5/ae_steps),iters=ae_steps)
                model.eval()
                logits,_ = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)
            sum1, cnt1, sumobj, cntobj = torch_dist_sum(local_rank, top1.sum, top1.cnt, objs.sum, objs.cnt)
            top1_acc = sum(sum1.float()) / sum(cnt1.float())
            obj__ = sum(sumobj.float()) / sum(cntobj.float())
            if rank==0:
              logging.info('valid (extra) eps=%.4e loss=%.4e top1-acc=%.4f ',extra_strengths[i], obj__, top1_acc)



    return top1_clean.avg, objs_clean.avg


def infer_clean(valid_queue, model, criterion):
    objs_clean = utils.AvgrageMeter()
    top1_clean = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    break_step=args.break_value

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if step==break_step:
                break
        
            input = Variable(input, requires_grad=False).cuda(non_blocking=True)
            target = Variable(target, requires_grad=False).cuda(non_blocking=True)

            logits,_ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs_clean.update(loss.data.item(), n)
            top1_clean.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)


    logging.info('valid loss and acc: %f %f', objs_clean.avg, top1_clean.avg)


    return top1_clean.avg, objs_clean.avg

def adjust_learning_rate(optimizer, epoch):
  """decrease the learning rate"""
  lr = args.learning_rate
  if epoch >= 99:
    lr = args.learning_rate * 0.1
  if epoch >= 149:
    lr = args.learning_rate * 0.01
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

if __name__ == '__main__':
  main() 

