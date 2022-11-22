import torch
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

from typing import Tuple

import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from timm.scheduler import create_scheduler
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from model import NetworkCIFAR as Network
import argparse
from util.dist_init import dist_init

import genotypes
import utils

model_name='E_ANE_minus'
est_name='est_cifar'
current_folder='.'

mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=10)




parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data_folder', help='location of the data corpus')
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
parser.add_argument('--init_channels', type=int, default=50, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='result_b2c4e40', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--port', type=int, default=23456, help='master port')
parser.add_argument('--break_value', type=int, default=-1, help='break value in train val loops')
parser.add_argument('--perturb_steps', type=int, default=10, help='steps to be perturbed in TRADES')

args = parser.parse_args()

rank, local_rank, world_size = dist_init(port=args.port)





class AN_estimator(nn.Module):
    def __init__(self,image_size,image_channel):
        super().__init__()
        self.f_kernelsize=3
        self.f_stride=1
        self.f_padding=1
        self.f_inchannels=7
        self.f_outchannels=3
        self.f_inter_channel=100

        self.b_kernelsize=self.f_kernelsize
        self.b_stride=self.f_stride
        self.b_padding=self.f_padding
        self.b_inchannels=self.f_inchannels
        self.b_outchannels=self.f_outchannels
        self.b_inter_channel=self.f_inchannels

        self.image_size=image_size
        self.image_channel=image_channel

        self.qk_channel=200
        self.scale = self.qk_channel ** -0.5


        self.forward_conv=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=self.f_inchannels,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=self.f_inter_channel,
                                    out_channels=self.f_outchannels,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU())

        self.backward_conv=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=self.f_inchannels,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=self.f_inter_channel,
                                    out_channels=self.f_outchannels,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU())

        self.q_conv=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=6,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=self.f_inter_channel,
                                    out_channels=3,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU())

        self.k_conv=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=self.image_channel*2,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=0),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=self.f_inter_channel,
                                    out_channels=self.qk_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=0),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=16))

        self.v_conv=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=self.image_channel*2,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=self.f_inter_channel,
                                    out_channels=self.f_outchannels,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU())

        self.norm_fm_encoder=torch.nn.Sequential(torch.nn.Linear(1,self.image_size**2),
                                                torch.nn.ReLU())

        self.new_fm = nn.Parameter(torch.rand(1,1,self.image_size,self.image_size))

        self.zero_image=torch.zeros(1,image_channel,image_size,image_size).cuda()

    def get_index(self, sequence, target):
        if target<sequence[0]:
            index1=None
            index2=0
            return index1,index2
        elif target>=sequence[-1]:
            index1=len(sequence)-1
            index2=None
            return index1,index2
        else:
            for i in range(len(sequence)-1):
                if target>=sequence[i] and target<sequence[i+1]:
                    return i,i+1

    def forward(self, source_noises,input_image,target_epsilons):
        '''
        source_noises: tensor, (bs,source_noise_num,image_channel,image_size,image_size),
                        the noises are ordered in terms of strength
        input_image: tensor, (bs,image_channel,image_size,image_size)
        target_epsilons: tensor, (b,n), where n is the number of targets to be generated
        label: tensor, (bs,image_channel,image_size,image_size)
        '''
        sn_shape=source_noises.shape
        source_noises_t=torch.reshape(source_noises,(sn_shape[0],sn_shape[1],sn_shape[2]*sn_shape[3]**2))
        source_noises_n=torch.norm(input=source_noises_t,dim=2,keepdim=True)
        source_noises_n=torch.reshape(source_noises_n,[sn_shape[0],sn_shape[1],1,1,1])
        normalized_noises=source_noises/source_noises_n
        source_noises_n=torch.squeeze(source_noises_n).float()

        strength_encodings=[]
        for i in range(source_noises_n.shape[1]):
            encoding=self.norm_fm_encoder(source_noises_n[:,i:(i+1)])
            encoding=torch.reshape(encoding,(encoding.shape[0],1,self.image_size,self.image_size))
            strength_encodings.append(encoding)

        target_epsilons=target_epsilons.float()
        target_strength_encodings=[]
        for i in range(target_epsilons.shape[1]):
            encoding=self.norm_fm_encoder(target_epsilons[:,i:(i+1)])
            encoding=torch.reshape(encoding,(encoding.shape[0],1,self.image_size,self.image_size))
            target_strength_encodings.append(encoding)

        #--------------forward rnn sequence-------------------------
        forward_outputs=[]
        for i in range(sn_shape[1]):
            current_inputs=[]
            if i==0:
                current_inputs.append(input_image)
            else:
                current_inputs.append(forward_outputs[i-1])
            current_inputs.append(strength_encodings[i])
            current_inputs.append(normalized_noises[:,i,:,:,:])

            input_tensor=torch.cat(current_inputs,1).float()
            output_tensor=self.forward_conv(input_tensor)
            forward_outputs.append(output_tensor)

        #--------------backward rnn sequence-------------------------
        backward_outputs=[]
        for i in range(sn_shape[1],0,-1):
            current_inputs=[]
            if i==sn_shape[1]:
                current_inputs.append(input_image)
            else:
                current_inputs.append(backward_outputs[0])
            current_inputs.append(strength_encodings[i-1])
            current_inputs.append(normalized_noises[:,i-1,:,:,:])
            input_tensor=torch.cat(current_inputs,1).float()
            output_tensor=self.backward_conv(input_tensor)
            backward_outputs.insert(0,output_tensor)

        noise_norm_np=source_noises_n.cpu().detach().numpy()
        target_norm_np=target_epsilons.cpu().detach().numpy()
        output1_list=[]
        output2_list=[]
        delta_d_matrix=[]
        for i in range(len(target_norm_np[0])):
            forward_batch=[]
            backward_batch=[]
            delta_d_batch=[]
            for j in range(len(noise_norm_np)):
                index1,index2=self.get_index(noise_norm_np[j],target_norm_np[j][i])
                if index1!=None:
                    forward_batch.append(torch.unsqueeze(forward_outputs[index1][j],0))
                else:
                    forward_batch.append(self.zero_image)
                if index2!=None:
                    backward_batch.append(torch.unsqueeze(backward_outputs[index2][j],0))
                else:
                    backward_batch.append(self.zero_image)

                if index1!=None and index2==None:
                    delta_d_batch.append(torch.unsqueeze(forward_outputs[index1][j],0)*(target_norm_np[j][i]/noise_norm_np[j][index1]))
                elif index2!=None and index1==None:
                    delta_d_batch.append(torch.unsqueeze(forward_outputs[index2][j],0)*(target_norm_np[j][i]/noise_norm_np[j][index2]))
                else:
                    noise1=torch.unsqueeze(forward_outputs[index1][j],0)
                    noise2=torch.unsqueeze(forward_outputs[index2][j],0)
                    sum=np.abs(target_norm_np[j][i]-noise_norm_np[j][index1])+np.abs(target_norm_np[j][i]-noise_norm_np[j][index2])
                    w1=np.abs(target_norm_np[j][i]-noise_norm_np[j][index1])/sum
                    w2=np.abs(target_norm_np[j][i]-noise_norm_np[j][index2])/sum
                    delta_d_batch.append(w1*noise1+w2*noise2)
            delta_d_batch=torch.cat(delta_d_batch,0)
            delta_d_matrix.append(torch.unsqueeze(delta_d_batch,1))

            forward_batch=torch.cat(forward_batch,0)
            backward_batch=torch.cat(backward_batch,0)

            feature_map1=[forward_batch]
            feature_map1.append(target_strength_encodings[i])
            feature_map1.append(self.new_fm.expand(len(noise_norm_np),self.image_channel,self.image_size,self.image_size))
            feature_map1=torch.cat(feature_map1,1)

            feature_map2=[backward_batch]
            feature_map2.append(target_strength_encodings[i])
            feature_map2.append(torch.reshape(delta_d_matrix[i],(len(noise_norm_np),self.image_channel,self.image_size,self.image_size)))
            feature_map2=torch.cat(feature_map2,1)

            out1=self.forward_conv(feature_map1)
            out2=self.forward_conv(feature_map2)

            output1_list.append(out1)
            output2_list.append(out2)

        delta_d_matrix=torch.cat(delta_d_matrix,1)

        query_list=[]
        key_list=[]
        value_list=[]
        for i in range(len(output1_list)):
            information=torch.cat([output1_list[i],output2_list[i]],1)
            information=torch.unsqueeze(self.q_conv(information),1)
            query_list.append(information)


        result=torch.cat(query_list,1)

        result_shape=result.shape
        result_n=torch.norm(torch.reshape(result,[result_shape[0],result_shape[1],result_shape[2]*result_shape[3]*result_shape[4]]),dim=2)
        result_n=torch.reshape(result_n,[result_shape[0],result_shape[1],1,1,1])
        target_norm_t=torch.reshape(target_epsilons,[result_shape[0],result_shape[1],1,1,1])
        result=(result/result_n)*target_norm_t

        return result


def test_estimator():
    bs=5
    image_channel=3
    image_size=32

    source_noise_num=6
    top_num=5
    norm_list=[]
    for j in range(bs):
        top=np.random.rand()*top_num
        temp_norms=[]
        for i in range(source_noise_num):
            temp_norms.insert(0,top)
            top=top-np.random.rand()*(top/2)
        norm_list.append(temp_norms)
    norm_list=np.asarray(norm_list)
    norm_list=np.reshape(norm_list,(norm_list.shape[0],norm_list.shape[1],1,1,1))
    norm_list=torch.tensor(norm_list)
    source_noises=torch.rand(bs,source_noise_num,image_channel,image_size,image_size)
    source_noises_t=torch.reshape(source_noises,(bs,source_noise_num,image_channel,image_size*image_size))
    source_noises_n=torch.unsqueeze(torch.norm(input=source_noises_t,dim=3,keepdim=True),-1)
    source_noises=(source_noises/source_noises_n)*norm_list

    input_image=torch.rand(bs,image_channel,image_size,image_size)

    target_epsilons=[]
    target_noise_num=4
    top_num=5
    target_epsilons=[]
    for j in range(bs):
        top=np.random.rand()*top_num
        temp_norms=[]
        for i in range(target_noise_num):
            temp_norms.insert(0,top)
            top=top-np.random.rand()*(top/2)
        target_epsilons.append(temp_norms)
    target_epsilons=np.asarray(target_epsilons)
    target_epsilons=torch.tensor(target_epsilons)

    label=torch.rand(bs,image_channel,image_size,image_size)


    noise_estimator=AN_estimator(image_size,image_channel)
    output=noise_estimator(source_noises,input_image,target_epsilons)

def pgd_attack(model, images, labels, loss, eps=0.3, alpha=2/255, iters=40):
    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs,_ = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).cuda()
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images=images.detach()

    noises=images-ori_images

    return images,noises

def form_data(noises0,noises1,noises2,noises3,batch_x,image_channel,image_size,batch_size,batch_y):

    noises0_=torch.unsqueeze(noises0,1)
    noises1_=torch.unsqueeze(noises1,1)
    noises2_=torch.unsqueeze(noises2,1)
    noises3_=torch.unsqueeze(noises3,1)

    source_noises0=torch.cat([noises1_,noises2_,noises3_],1)
    input_image0=batch_x
    target_epsilons0=torch.norm(torch.reshape(noises0,[noises0.shape[0],image_channel*image_size*image_size]),dim=1,keepdim=True)
    label0=noises0

    source_noises1=torch.cat([noises0_,noises2_,noises3_],1)
    input_image1=batch_x
    target_epsilons1=torch.norm(torch.reshape(noises1,[noises1.shape[0],image_channel*image_size*image_size]),dim=1,keepdim=True)
    label1=noises1

    source_noises2=torch.cat([noises0_,noises1_,noises3_],1)
    input_image2=batch_x
    target_epsilons2=torch.norm(torch.reshape(noises2,[noises2.shape[0],image_channel*image_size*image_size]),dim=1,keepdim=True)
    label2=noises2

    source_noises3=torch.cat([noises0_,noises1_,noises2_],1)
    input_image3=batch_x
    target_epsilons3=torch.norm(torch.reshape(noises3,[noises3.shape[0],image_channel*image_size*image_size]),dim=1,keepdim=True)
    label3=noises3

    source_noises_combined=torch.cat([source_noises0.cuda(),source_noises1.cuda(),source_noises2.cuda(),source_noises3.cuda()],0)
    input_image_combined=torch.cat([input_image0.cuda(),input_image1.cuda(),input_image2.cuda(),input_image3.cuda()],0)
    target_epsilons_combined=torch.cat([target_epsilons0.cuda(),target_epsilons1.cuda(),target_epsilons2.cuda(),target_epsilons3.cuda()],0)
    label_combined=torch.cat([label0.cuda(),label1.cuda(),label2.cuda(),label3.cuda()],0)
    batch_y_combined=torch.cat([batch_y.cuda(),batch_y.cuda(),batch_y.cuda(),batch_y.cuda()],0)

    order=np.arange(0,batch_size*4,1)
    np.random.shuffle(order)

    order_list=[]
    for i in range(4):
        order_=order[i*batch_size:(i+1)*batch_size]
        order_list.append(order_)
    order_list=torch.tensor(np.asarray(order_list)).cuda()

    source_noise_list=[]
    input_image_list=[]
    target_epsilons_list=[]
    label_list=[]
    batch_y_list=[]
    for i in range(len(order_list)):
        source_noise_list.append(source_noises_combined[order_list[i]])
        input_image_list.append(input_image_combined[order_list[i]])
        target_epsilons_list.append(target_epsilons_combined[order_list[i]])
        label_list.append(label_combined[order_list[i]])
        batch_y_list.append(batch_y_combined[order_list[i]])

    return source_noise_list,input_image_list,target_epsilons_list,label_list,order_list,batch_y_list

def increment_model_index(i):
    if i+1==3:
        return 0
    else:
        return i+1

def train_estimator():
    image_size=32
    num_classes=10
    epochs=40
    batch_size=32
    accumulate_num=1
    image_channel=3

    pgd_alpha=0.03
    pgd_step=6

    val_break=50
    train_break=300
    print_step=25

    random_constant=0.15

    genotype_list=['cna_r','cna_m','cna_n',
                    'cna_m_1','cna_r_minus','cna_r',
                    'cna_m',]
    init_channel_list=[45,45,50,
                        45,42,45,
                        45]
    layer_list=[20,20,20,
                20,20,20,
                20]
    state_dict_list=['weight1.pth.tar',
                    'weight2.pth.tar',
                    'weight3.pth.tar',
                    'weight4.pth.tar',
                    'weight5.pth.tar',
                    'weight6.pth.tar',
                    'weight7.pth.tar']
    step_counter=0
    step_limit=100
    model_index=0

    genotype = eval("genotypes.%s" % genotype_list[model_index])
    model = Network(init_channel_list[model_index], 10, layer_list[model_index], False, genotype)
    model = model.cuda()
    model.drop_path_prob=0
    model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False)
    checkpoint = torch.load(state_dict_list[model_index],map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model_index=increment_model_index(model_index)

    noise_estimator=AN_estimator(image_size,image_channel)
    noise_estimator=noise_estimator.cuda()

    optimizer_ne = torch.optim.AdamW(noise_estimator.parameters(), lr=0.0005, weight_decay=0.0001)
    lambda1=lambda epoch:(epoch/4000) if epoch<4000 else 0.5*(math.cos((epoch-4000)/(100*1000-4000)*math.pi)+1)
    scheduler_ne=optim.lr_scheduler.LambdaLR(optimizer_ne,lr_lambda=lambda1)


    transform_train, transforms_ = utils._data_transforms_cifar10_eval(args)

    dataset=torchvision.datasets.CIFAR10(root='datafolder',transform=transform_train,download=True, train=True)
    test_data=torchvision.datasets.CIFAR10(root='datafolder',transform=transforms_,download=True, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size
    )
    ce_loss = torch.nn.CrossEntropyLoss()


    best_val_acc=0


    acc_adv_meam=0
    acc_est_mean=0
    for i in range(len(genotype_list)):
        genotype = eval("genotypes.%s" % genotype_list[i])
        model = Network(init_channel_list[i], 10, layer_list[i], False, genotype)
        model = model.cuda()
        model.drop_path_prob=0
        model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False)
        checkpoint = torch.load(state_dict_list[i],map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
            
        val_data_count=0
        val_acc_sum=0
        ae_data_count=0
        ae_acc_sum=0
        est_data_count=0
        est_acc_sum=0
        for stepv, (batch_xv,batch_yv) in enumerate(test_loader):
            if stepv%print_step==0:
                print('stepv: ',stepv)
            if stepv==val_break or batch_xv.shape[0]!=batch_size:
                break
            batch_xv=batch_xv.cuda()
            batch_yv=batch_yv.cuda()

            images0,noises0=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=0.03, alpha=0.03*2.5/7, iters=7)
            images1,noises1=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=0.06, alpha=0.06*2.5/7, iters=7)
            images2,noises2=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=0.09, alpha=0.09*2.5/7, iters=7)
            eps_=np.random.rand()*random_constant
            images3,noises3=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=eps_, alpha=eps_*2.5/7, iters=7)

            ae_list=[images0,images1,images2,images3]

            source_noise_list,input_image_list,target_epsilons_list,label_list,order_list,batch_yv_list=form_data(noises0,noises1,noises2,noises3,batch_xv,image_channel,image_size,batch_size,batch_yv)

            val_data_count+=batch_xv.shape[0]

            pred_ae,_=model(batch_xv)
            pred_aet=pred_ae.cpu()
            pred_ae_np=pred_aet.detach().numpy()
            batch_yt=batch_yv.cpu()
            label_np=batch_yt.detach().numpy()
            for j in range(len(pred_ae_np)):
                if np.argmax(pred_ae_np[j])==label_np[j]:
                    val_acc_sum+=1

            for i in range(len(ae_list)):
                ae_data_count+=ae_list[i].shape[0]
                pred_ae,_=model(ae_list[i])
                pred_aet=pred_ae.cpu()
                pred_ae_np=pred_aet.detach().numpy()
                batch_yt=batch_yv.cpu()
                label_np=batch_yt.detach().numpy()
                for j in range(len(pred_ae_np)):
                    if np.argmax(pred_ae_np[j])==label_np[j]:
                        ae_acc_sum+=1

            for i in range(len(order_list)):
                est_data_count+=source_noise_list[i].shape[0]
                n_estimate=noise_estimator(source_noise_list[i],input_image_list[i],target_epsilons_list[i])
                pred_ae,_=model(torch.squeeze(n_estimate,1)+input_image_list[i])
                pred_aet=pred_ae.cpu()
                pred_ae_np=pred_aet.detach().numpy()
                batch_yt=batch_yv_list[i].cpu()
                label_np=batch_yt.detach().numpy()
                for j in range(len(pred_ae_np)):
                    if np.argmax(pred_ae_np[j])==label_np[j]:
                        est_acc_sum+=1

        print('genotype: ',genotype)
        print('clean accuracy: ',val_acc_sum/val_data_count)
        print('ae accuracy: ',ae_acc_sum/ae_data_count)
        print('est accuracy: ',est_acc_sum/est_data_count)

        acc_adv_meam+=ae_acc_sum/ae_data_count
        acc_est_mean+=est_acc_sum/est_data_count
    
    print(' val acc adv (mean): ',acc_adv_meam/len(genotype_list),
        ' val acc est(mean): ',acc_est_mean/len(genotype_list))
    with open(current_folder+'/'+model_name+'_'+est_name+'_stdout'+'.txt', "a") as std_out:
            std_out.write(
            ' val acc adv (mean): '+str(acc_adv_meam/len(genotype_list))+
            ' val acc est(mean): '+str(acc_est_mean/len(genotype_list))+'\n')
            std_out.write('\n')
            std_out.close()

    with open(current_folder+'/'+model_name+'_'+est_name+'_stdout'+'.txt', "a") as std_out:
        std_out.write(
        ' clean accuracy: '+str(val_acc_sum/val_data_count)+
        ' ae accuracy: '+str(ae_acc_sum/ae_data_count)+
        ' est accuracy: '+str(est_acc_sum/est_data_count)+'\n')
        std_out.write('\n')
        std_out.close()


    for epoch in range(epochs):
        accumulated_count=0

        model.train()
        for step, (batch_x_,batch_y_) in enumerate(train_loader):
            step_counter+=1
            if step_counter==step_limit:
                genotype = eval("genotypes.%s" % genotype_list[model_index])
                model = Network(init_channel_list[model_index], 10, layer_list[model_index], False, genotype)
                model = model.cuda()
                model.drop_path_prob=0
                model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False)
                checkpoint = torch.load(state_dict_list[model_index],map_location='cuda')
                model.load_state_dict(checkpoint['state_dict'])
                model_index=increment_model_index(model_index)
            
            if step%print_step==0:
                print('step: ',step)
            if step==train_break or batch_x_.shape[0]!=batch_size:
                break

            if accumulated_count==0:
                optimizer_ne.zero_grad()

            if batch_x_.shape[0]!=batch_size:
                break
            noise_estimator.train()
            batch_x=batch_x_.cuda()
            batch_y=batch_y_.cuda()

            images0,noises0=pgd_attack(model, batch_x, batch_y, ce_loss, eps=0.03, alpha=0.03*2.5/7, iters=7)
            images1,noises1=pgd_attack(model, batch_x, batch_y, ce_loss, eps=0.06, alpha=0.06*2.5/7, iters=7)
            images2,noises2=pgd_attack(model, batch_x, batch_y, ce_loss, eps=0.09, alpha=0.09*2.5/7, iters=7)
            eps_=np.random.rand()*random_constant
            images3,noises3=pgd_attack(model, batch_x, batch_y, ce_loss, eps=eps_, alpha=eps_*2.5/7, iters=7)

            ae_list=[images0,images1,images2,images3]

            source_noise_list,input_image_list,target_epsilons_list,label_list,order_list,batch_y_list=form_data(noises0,noises1,noises2,noises3,batch_x,image_channel,image_size,batch_size,batch_y)

            for i in range(len(order_list)):
                n_estimate=noise_estimator(source_noise_list[i],input_image_list[i],target_epsilons_list[i])
                temp=torch.squeeze(n_estimate,1)-label_list[i]
                loss=temp*temp
                loss=torch.mean(loss)
                loss.backward()
                optimizer_ne.step()
            scheduler_ne.step()



        train_data_count=0
        train_loss_sum=0
        train_adv_acc_sum=0
        train_est_acc_sum=0

        for i in range(len(order_list)):
            input_images=input_image_list[i]
            n_estimate=noise_estimator(source_noise_list[i],input_images,target_epsilons_list[i])

            train_data_count+=n_estimate.shape[0]

            current_label=label_list[i]
            temp=torch.squeeze(n_estimate,1)-current_label
            loss=temp*temp
            loss=torch.mean(loss)
            train_loss_sum+=loss.cpu().detach().numpy()

            pred_ae,_=model(ae_list[i])
            pred_aet=pred_ae.cpu()
            pred_ae_np=pred_aet.detach().numpy()
            batch_yt=batch_y.cpu()
            label_np=batch_yt.detach().numpy()
            for j in range(len(pred_ae_np)):
                if np.argmax(pred_ae_np[j])==label_np[j]:
                    train_adv_acc_sum+=1

            pred_ae,_=model(torch.squeeze(n_estimate,1)+input_images)
            pred_aet=pred_ae.cpu()
            pred_ae_np=pred_aet.detach().numpy()
            batch_yt=batch_y_list[i].cpu()
            label_np=batch_yt.detach().numpy()
            for j in range(len(pred_ae_np)):
                if np.argmax(pred_ae_np[j])==label_np[j]:
                    train_est_acc_sum+=1

        acc_adv_meam=0
        acc_est_mean=0

        for u in range(len(genotype_list)):
            genotype = eval("genotypes.%s" % genotype_list[u])
            model = Network(init_channel_list[u], 10, layer_list[u], False, genotype)
            model = model.cuda()
            model.drop_path_prob=0
            model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False)
            checkpoint = torch.load(state_dict_list[u],map_location='cuda')
            model.load_state_dict(checkpoint['state_dict'])

            #validation result
            val_data_count=0
            val_loss_sum=0
            val_adv_acc_sum=0
            val_est_acc_sum=0
            for stepv, (batch_xv,batch_yv) in enumerate(test_loader):
                if stepv%print_step==0:
                    print('stepv: ',stepv)
                if stepv==val_break or batch_xv.shape[0]!=batch_size:
                    break
                batch_xv=batch_xv.cuda()
                batch_yv=batch_yv.cuda()
                images0,noises0=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=0.03, alpha=0.03*2.5/7, iters=7)
                images1,noises1=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=0.06, alpha=0.06*2.5/7, iters=7)
                images2,noises2=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=0.09, alpha=0.09*2.5/7, iters=7)
                eps_=np.random.rand()*random_constant
                images3,noises3=pgd_attack(model, batch_xv, batch_yv, ce_loss, eps=eps_, alpha=eps_*2.5/7, iters=7)

                ae_list=[images0,images1,images2,images3]

                source_noise_list,input_image_list,target_epsilons_list,label_list,order_list,batch_y_list=form_data(noises0,noises1,noises2,noises3,batch_xv,image_channel,image_size,batch_size,batch_yv)

                for i in range(len(order_list)):
                    input_images=input_image_list[i].cuda()
                    n_estimate=noise_estimator(source_noise_list[i],input_images,target_epsilons_list[i])

                    val_data_count+=n_estimate.shape[0]

                    current_label=label_list[i]
                    temp=torch.squeeze(n_estimate,1)-current_label
                    loss=temp*temp
                    loss=torch.mean(loss)
                    val_loss_sum+=loss.cpu().detach().numpy()

                    pred_ae,_=model(ae_list[i])
                    pred_aet=pred_ae.cpu()
                    pred_ae_np=pred_aet.detach().numpy()
                    batch_yt=batch_yv.cpu()
                    label_np=batch_yt.detach().numpy()
                    for j in range(len(pred_ae_np)):
                        if np.argmax(pred_ae_np[j])==label_np[j]:
                            val_adv_acc_sum+=1

                    pred_ae,_=model(torch.squeeze(n_estimate,1)+input_images)
                    pred_aet=pred_ae.cpu()
                    pred_ae_np=pred_aet.detach().numpy()
                    batch_yt=batch_y_list[i].cpu()
                    label_np=batch_yt.detach().numpy()
                    for j in range(len(pred_ae_np)):
                        if np.argmax(pred_ae_np[j])==label_np[j]:
                            val_est_acc_sum+=1



            print('epoch: ',epoch,
                'genotype: ',genotype_list[u])

            print(' val loss: ',val_loss_sum/val_data_count,
                ' val acc adv: ',val_adv_acc_sum/val_data_count,
                ' val acc est: ',val_est_acc_sum/val_data_count)
            
            acc_adv_meam+=val_adv_acc_sum/val_data_count
            acc_est_mean+=val_est_acc_sum/val_data_count

            with open(current_folder+'/'+model_name+'_'+est_name+'_stdout'+'.txt', "a") as std_out:
                std_out.write('epoch: '+str(epoch)+
                ' genotype: '+str(genotype_list[u])+'\n')
                std_out.write(
                ' val loss: '+str(val_loss_sum/val_data_count)+
                ' val acc adv: '+str(val_adv_acc_sum/val_data_count)+
                ' val acc est: '+str(val_est_acc_sum/val_data_count)+'\n')
                std_out.write('\n')
                std_out.close()
            
        print(' val acc adv (mean): ',acc_adv_meam/len(genotype_list),
            ' val acc est(mean): ',acc_est_mean/len(genotype_list))
        with open(current_folder+'/'+model_name+'_'+est_name+'_stdout'+'.txt', "a") as std_out:
                std_out.write(
                ' val acc adv (mean): '+str(acc_adv_meam/len(genotype_list))+
                ' val acc est(mean): '+str(acc_est_mean/len(genotype_list))+'\n')
                std_out.write('\n')
                std_out.close()
        torch.save(model.state_dict(), current_folder+'/'+model_name+'/'+model_name+'_'+est_name)

if __name__=='__main__':

    std_out=open(current_folder+'/'+model_name+'_'+est_name+'_stdout'+'.txt','w+')
    std_out.close()
    if not os.path.exists(current_folder+'/'+model_name):
        os.mkdir(current_folder+'/'+model_name)


    train_estimator()




