import argparse
import glob
import logging
import os
from pickle import NONE
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from scipy.stats import kendalltau
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10

import utils
from module.architect3_9 import Architect
from module.estimator.estimator import Estimator, PredictorForGraph
from module.estimator.gnn.decoder import LinearDecoder
from module.estimator.gnn.encoder import GINEncoder
from module.estimator.gnn.gae import GAEExtractor
from module.estimator.gnn.loss import ReconstructedLoss
from module.estimator.memory import Memory
from module.estimator.population import Population
from module.estimator.predictor2 import Predictor, weighted_loss
from module.estimator.utils import GraphPreprocessor
from module.model_search import Network
from utils import gumbel_like, gpu_usage, DimensionImportanceWeight

from module.estimator.AE_memory import AE_Memory


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

from TRADES import trades_loss

current_folder='.'
model_name='C_NAS_search'
est_name='est'
ae_val_break=None

CIFAR_CLASSES = 10

class AN_estimator(nn.Module):
    def __init__(self,image_size,image_channel):
        super().__init__()
        self.image_size=image_size
        self.image_channel=image_channel

        self.layer1_k_size=5
        self.layer1_stride=self.layer1_k_size-1
        self.layer1_channel=10

        self.layer1_out_c=90

        self.strength_encoding_length=20
        self.hidden_dim=self.layer1_out_c

        self.n_preproc_dim=50

        self.noise_preproc=torch.nn.Sequential(torch.nn.Conv2d(in_channels=image_channel,
                                            out_channels=100,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=100,
                                            out_channels=3,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1))

        self.forward_conv1=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=image_channel,
                                    out_channels=self.layer1_channel,
                                    kernel_size=self.layer1_k_size,
                                    stride=self.layer1_stride,
                                    padding=0),
                            torch.nn.ReLU())
        self.forward_pool1=torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0)

        self.backward_conv1=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=image_channel,
                                    out_channels=self.layer1_channel,
                                    kernel_size=self.layer1_k_size,
                                    stride=self.layer1_stride,
                                    padding=0),
                            torch.nn.ReLU())
        self.backward_pool1=torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0)

        self.input_conv=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=image_channel,
                                    out_channels=self.layer1_channel,
                                    kernel_size=self.layer1_k_size,
                                    stride=self.layer1_stride,
                                    padding=0),
                            torch.nn.ReLU())
        self.input_pool=torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0)


        self.strength_encoder=torch.nn.Sequential(torch.nn.Linear(1,self.strength_encoding_length),
                                                torch.nn.ReLU())

        self.forward_hidden=torch.nn.Sequential(torch.nn.Linear(self.strength_encoding_length+self.layer1_out_c+self.hidden_dim,self.hidden_dim),
                                                torch.nn.ReLU())
        self.backward_hidden=torch.nn.Sequential(torch.nn.Linear(self.strength_encoding_length+self.layer1_out_c+self.hidden_dim,self.hidden_dim),
                                                torch.nn.ReLU())


        self.forward_query=torch.nn.Sequential(torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                                torch.nn.ReLU())
        self.backward_query=torch.nn.Sequential(torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                                torch.nn.ReLU())

        self.forward_value=torch.nn.Sequential(torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                                torch.nn.ReLU())
        self.backward_value=torch.nn.Sequential(torch.nn.Linear(self.hidden_dim,self.hidden_dim),
                                                torch.nn.ReLU())

        self.zero_image=torch.zeros(1,1,self.hidden_dim).cuda()
        self.zero_noise=torch.zeros(1,1,self.image_channel,self.image_size,self.image_size)

    def get_index(self, sequence, target,none_index):
        if target<sequence[0]:
            index1=none_index
            index2=0
            return index1,index2
        elif target>=sequence[-1]:
            index1=len(sequence)-1
            index2=none_index
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

        result: tensor:(bs,n,image_channel,image_size,image_size)
        '''

        sn_shape=source_noises.shape
        source_noises_t=torch.reshape(source_noises,(sn_shape[0],sn_shape[1],sn_shape[2]*sn_shape[3]**2))
        source_noises_n=torch.norm(input=source_noises_t,dim=2,keepdim=True)
        source_noises_n=torch.reshape(source_noises_n,[sn_shape[0],sn_shape[1],1,1,1])
        normalized_noises=source_noises/source_noises_n
        normalized_noises=normalized_noises.float()
        source_noises_n=torch.squeeze(source_noises_n).float()


        normalized_noises_temp=normalized_noises
        normalized_noises=torch.reshape(normalized_noises,[sn_shape[0]*sn_shape[1],sn_shape[2],sn_shape[3],sn_shape[4]])
        normalized_noises=self.noise_preproc(normalized_noises)
        normalized_noises=torch.reshape(normalized_noises,sn_shape)+normalized_noises_temp

        #feed through layer 1
        reshaped_noises=torch.reshape(normalized_noises,(sn_shape[0]*sn_shape[1],sn_shape[2],sn_shape[3],sn_shape[4]))

        forward_1=self.forward_conv1(reshaped_noises.float())
        forward_1=self.forward_pool1(forward_1)
        forward_1=torch.reshape(forward_1,(sn_shape[0],sn_shape[1],forward_1.shape[1]*forward_1.shape[2]*forward_1.shape[3]))
        forward_1_shape=forward_1.shape

        backward_1=self.backward_conv1(reshaped_noises.float())
        backward_1=self.backward_pool1(backward_1)
        backward_1=torch.reshape(backward_1,(sn_shape[0],sn_shape[1],backward_1.shape[1]*backward_1.shape[2]*backward_1.shape[3]))
        backward_1_shape=backward_1.shape

        #generate strength encodings
        strength_encoding=torch.reshape(source_noises_n,(sn_shape[0],sn_shape[1],1)).float()
        strength_encoding=self.strength_encoder(strength_encoding)

        #concat
        forward_1=torch.cat([forward_1,strength_encoding],2)
        backward_1=torch.cat([backward_1,strength_encoding],2)
        forward_1_shape=forward_1.shape
        backward_1_shape=backward_1.shape




        #input encoding
        input_encoding=self.input_conv(input_image.float())
        input_encoding=self.input_pool(input_encoding)
        input_encoding=torch.reshape(input_encoding,(sn_shape[0],input_encoding.shape[1]*input_encoding.shape[2]*input_encoding.shape[3]))

        #forward backward RNN
        expanded_zero=self.zero_image.expand(forward_1_shape[0],self.zero_image.shape[1],self.zero_image.shape[2]).cuda()

        forward_hidden_states=[]
        for i in range(forward_1_shape[1]):
            if i==0:
                current_hidden=torch.cat([forward_1[:,i,:],input_encoding],1)
            else:
                current_hidden=torch.cat([forward_1[:,i,:],forward_hidden_states[-1][:,0,:]],1)
            next_hidden=self.forward_hidden(current_hidden)
            forward_hidden_states.append(torch.unsqueeze(next_hidden,1))
        forward_hidden_states=torch.cat(forward_hidden_states,1)
        forward_hidden_states=torch.cat([forward_hidden_states,expanded_zero],1)

        backward_hidden_states=[]
        for i in range(backward_1_shape[1]):
            if i==0:
                current_hidden=torch.cat([backward_1[:,i,:],input_encoding],1)
            else:
                current_hidden=torch.cat([backward_1[:,i,:],backward_hidden_states[0][:,0,:]],1)
            next_hidden=self.backward_hidden(torch.unsqueeze(current_hidden,1))
            backward_hidden_states.insert(0,next_hidden)
        backward_hidden_states=torch.cat(backward_hidden_states,1)
        backward_hidden_states=torch.cat([backward_hidden_states,expanded_zero],1)


        #form query
        noise_norm_np=source_noises_n.cpu().detach().numpy()
        target_norm_np=target_epsilons.cpu().detach().numpy()

        output1_list=[]
        output2_list=[]
        delta_d_matrix=[]
        none_index=backward_hidden_states.shape[1]-1
        index1_matrix=[]
        index2_matrix=[]
        w1_matrix=[]
        w2_matrix=[]
        for j in range(len(noise_norm_np)):
            index1_list=[]
            index2_list=[]
            w1_list=[]
            w2_list=[]
            for i in range(len(target_norm_np[j])):
                index1,index2=self.get_index(noise_norm_np[j],target_norm_np[j][i],none_index)
                index1_list.append(index1)
                index2_list.append(index2)
                if index1!=none_index and index2==none_index:
                    w1=0
                    w2=target_norm_np[j][i]/noise_norm_np[j][index1]
                elif index2!=none_index and index1==none_index:
                    w1=target_norm_np[j][i]/noise_norm_np[j][index2]
                    w2=0
                else:
                    sum=np.abs(target_norm_np[j][i]-noise_norm_np[j][index1])+np.abs(target_norm_np[j][i]-noise_norm_np[j][index2])
                    w1=np.abs(target_norm_np[j][i]-noise_norm_np[j][index1])/sum
                    w2=np.abs(target_norm_np[j][i]-noise_norm_np[j][index2])/sum
                w1_list.append(w1)
                w2_list.append(w2)
            index1_matrix.append(index1_list)
            index2_matrix.append(index2_list)
            w1_matrix.append(w1_list)
            w2_matrix.append(w2_list)

        index1_matrix=torch.tensor(np.asarray(index1_matrix)).type(torch.LongTensor).cuda()
        index2_matrix=torch.tensor(np.asarray(index2_matrix)).type(torch.LongTensor).cuda()
        w1_matrix=torch.tensor(np.asarray(w1_matrix)).float().cuda()
        w2_matrix=torch.tensor(np.asarray(w2_matrix)).float().cuda()

        w1_matrix=torch.unsqueeze(w1_matrix,2)
        w2_matrix=torch.unsqueeze(w2_matrix,2)

        index1_matrix_=torch.reshape(index1_matrix,(index1_matrix.shape[0],index1_matrix.shape[1],1,1,1))
        index1_matrix_=index1_matrix_.expand(index1_matrix.shape[0],index1_matrix.shape[1],self.image_channel,self.image_size,self.image_size)
        index2_matrix_=torch.reshape(index2_matrix,(index2_matrix.shape[0],index2_matrix.shape[1],1,1,1))
        index2_matrix_=index2_matrix_.expand(index2_matrix.shape[0],index2_matrix.shape[1],self.image_channel,self.image_size,self.image_size)

        nn_shape=normalized_noises.shape
        expanded_zero_noise=self.zero_noise.expand(nn_shape[0],1,nn_shape[2],nn_shape[3],nn_shape[4]).cuda()
        normalized_noises_=torch.cat([normalized_noises,expanded_zero_noise],1)

        w1_matrix_=torch.reshape(w1_matrix,[w1_matrix.shape[0],w1_matrix.shape[1],1,1,1])
        w2_matrix_=torch.reshape(w2_matrix,[w2_matrix.shape[0],w2_matrix.shape[1],1,1,1])

        gathered_noise1=torch.gather(normalized_noises_,1,index1_matrix_)
        gathered_noise1_weighted=gathered_noise1*w1_matrix_
        gathered_noise2=torch.gather(normalized_noises_,1,index2_matrix_)
        gathered_noise2_weighted=gathered_noise2*w2_matrix_

        delta_d_matrix=gathered_noise1_weighted+gathered_noise2_weighted

        index1_matrix=torch.unsqueeze(index1_matrix,2)
        index1_matrix=index1_matrix.expand(index1_matrix.shape[0],index1_matrix.shape[1],self.hidden_dim)
        index2_matrix=torch.unsqueeze(index2_matrix,2)
        index2_matrix=index2_matrix.expand(index2_matrix.shape[0],index2_matrix.shape[1],self.hidden_dim)

        gathered_forward_hidden=torch.gather(forward_hidden_states,1, index1_matrix)
        weighted_gathered_forward_hidden=w1_matrix*gathered_forward_hidden
        gathered_backward_hidden=torch.gather(backward_hidden_states,1, index2_matrix)
        weighted_gathered_backward_hidden=w2_matrix*gathered_backward_hidden

        delta_matrix=weighted_gathered_forward_hidden+weighted_gathered_backward_hidden


        target_n_enc=self.strength_encoder(torch.unsqueeze(target_epsilons.float(),2))

        query_forward=torch.cat([delta_matrix,target_n_enc,gathered_forward_hidden],2)
        query_forward=self.forward_hidden(query_forward)
        query_forward=self.forward_query(query_forward)

        query_backward=torch.cat([delta_matrix,target_n_enc,gathered_backward_hidden],2)
        query_backward=self.backward_hidden(query_backward)
        query_backward=self.backward_query(query_backward)

        query_forward=self.forward_query(query_forward)
        query_backward=self.forward_query(query_backward)
        query=torch.cat([query_forward,query_backward],2)
        query=torch.transpose(query,1,2)

        value_forward=self.forward_value(forward_hidden_states[:,:-1,:])
        value_backward=self.forward_value(backward_hidden_states[:,:-1,:])
        value=torch.cat([value_forward,value_backward],2)

        attention_map=torch.matmul(value,query)

        t_shape=normalized_noises.shape
        temp_noises=torch.reshape(normalized_noises,[t_shape[0],t_shape[1],t_shape[2]*t_shape[3]*t_shape[4]])
        temp_noises=torch.transpose(temp_noises,1,2)
        result=torch.matmul(temp_noises,attention_map)

        r_shape=result.shape
        delta_dd=torch.reshape(result,[r_shape[0],r_shape[2],t_shape[2],t_shape[3],t_shape[4]])

        result=delta_dd+delta_d_matrix
        result_shape=result.shape
        result_n=torch.norm(torch.reshape(result,[result_shape[0],result_shape[1],result_shape[2]*result_shape[3]*result_shape[4]]),dim=2)
        result_n=torch.reshape(result_n,[result_shape[0],result_shape[1],1,1,1])
        target_norm_t=torch.reshape(target_epsilons,[result_shape[0],result_shape[1],1,1,1])
        result=(result/result_n)*target_norm_t

        return result.float()



class AN_estimator_plus(nn.Module):
    def __init__(self,image_size,image_channel):
        super().__init__()

        '''
        When implementing this version of AN-Estimator, the "Depth-Wise Separable Convolution"
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks" has been leveraged to improve
        performance.
        '''

        self.f_kernelsize=3
        self.f_stride=1
        self.f_padding=1
        self.f_inchannels=7
        self.f_outchannels=3
        self.f_inter_channel=300

        self.b_kernelsize=self.f_kernelsize
        self.b_stride=self.f_stride
        self.b_padding=self.f_padding
        self.b_inchannels=self.f_inchannels
        self.b_outchannels=self.f_outchannels
        self.b_inter_channel=self.f_inchannels

        self.image_size=image_size
        self.image_channel=image_channel

        self.qk_channel=50
        self.scale = self.qk_channel ** -0.5

        self.forward_conv=torch.nn.Sequential(
                            nn.BatchNorm2d(self.f_inchannels),
                            torch.nn.Conv2d(in_channels=self.f_inchannels,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),

                            # pw
                            nn.Conv2d(self.f_inter_channel, self.f_outchannels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),
                            nn.ReLU6(inplace=True),
                            # dw
                            nn.Conv2d(self.f_outchannels, self.f_outchannels, self.f_kernelsize, self.f_stride, self.f_padding, groups=self.f_outchannels, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),
                            nn.ReLU6(inplace=True),
                            # pw-linear
                            nn.Conv2d(self.f_outchannels, self.f_outchannels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.f_outchannels)

                            )

        self.backward_conv=torch.nn.Sequential(
                            nn.BatchNorm2d(self.f_inchannels),
                            torch.nn.Conv2d(in_channels=self.f_inchannels,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),
                            # pw
                            nn.Conv2d(self.f_inter_channel, self.f_outchannels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),
                            nn.ReLU6(inplace=True),
                            # dw
                            nn.Conv2d(self.f_outchannels, self.f_outchannels, self.f_kernelsize, self.f_stride, self.f_padding, groups=self.f_outchannels, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),
                            nn.ReLU6(inplace=True),
                            # pw-linear
                            nn.Conv2d(self.f_outchannels, self.f_outchannels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.f_outchannels)
                            )

        self.q_conv=torch.nn.Sequential(
                            nn.BatchNorm2d(self.image_channel*3),
                            torch.nn.Conv2d(in_channels=self.image_channel*3,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=0),
                            torch.nn.ReLU(),
                            # pw
                            nn.Conv2d(self.f_inter_channel, self.qk_channel, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.qk_channel),
                            nn.ReLU6(inplace=True),
                            # dw
                            nn.Conv2d(self.qk_channel, self.qk_channel, self.f_kernelsize, self.f_stride, 0, groups=self.qk_channel, bias=False),
                            nn.BatchNorm2d(self.qk_channel),
                            nn.ReLU6(inplace=True),
                            # pw-linear
                            nn.Conv2d(self.qk_channel, self.qk_channel, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.qk_channel),
                            torch.nn.MaxPool2d(kernel_size=16)
                            )



        self.k_conv=torch.nn.Sequential(
                            nn.BatchNorm2d(self.image_channel*2),
                            torch.nn.Conv2d(in_channels=self.image_channel*2,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=0),
                            torch.nn.ReLU(),
                            # pw
                            nn.Conv2d(self.f_inter_channel, self.qk_channel, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.qk_channel),
                            nn.ReLU6(inplace=True),
                            # dw
                            nn.Conv2d(self.qk_channel, self.qk_channel, self.f_kernelsize, self.f_stride, 0, groups=self.qk_channel, bias=False),
                            nn.BatchNorm2d(self.qk_channel),
                            nn.ReLU6(inplace=True),
                            # pw-linear
                            nn.Conv2d(self.qk_channel, self.qk_channel, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.qk_channel),
                            torch.nn.MaxPool2d(kernel_size=16))

        self.v_conv=torch.nn.Sequential(
                            nn.BatchNorm2d(self.image_channel*2),
                            torch.nn.Conv2d(in_channels=self.image_channel*2,
                                    out_channels=self.f_inter_channel,
                                    kernel_size=self.f_kernelsize,
                                    stride=self.f_stride,
                                    padding=self.f_padding),
                            torch.nn.ReLU(),

                            # pw
                            nn.Conv2d(self.f_inter_channel, self.f_outchannels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),
                            nn.ReLU6(inplace=True),
                            # dw
                            nn.Conv2d(self.f_outchannels, self.f_outchannels, self.f_kernelsize, self.f_stride, self.f_padding, groups=self.f_outchannels, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),
                            nn.ReLU6(inplace=True),
                            # pw-linear
                            nn.Conv2d(self.f_outchannels, self.f_outchannels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.f_outchannels),

                            )

        self.norm_fm_encoder=torch.nn.Sequential(
                                                torch.nn.Linear(1,self.image_size**2),
                                                torch.nn.ReLU())

        self.new_fm = nn.Parameter(torch.rand(1,1,self.image_size,self.image_size))

        self.zero_image=torch.zeros(1,image_channel,image_size,image_size).cuda()

    def get_index(self, sequence, target,none_index):
        if target<sequence[0]:
            index1=none_index
            index2=0
            return index1,index2
        elif target>=sequence[-1]:
            index1=len(sequence)-1
            index2=none_index
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

        ts=source_noises_n.shape
        source_noises_nr=torch.reshape(source_noises_n,(ts[0]*ts[1],1))
        strength_encodings=self.norm_fm_encoder(source_noises_nr)
        strength_encodings=torch.reshape(strength_encodings,(ts[0],ts[1],self.image_size,self.image_size))

        target_epsilons=target_epsilons.float()
        tes=target_epsilons.shape
        target_epsilons_r=torch.reshape(target_epsilons,(tes[0]*tes[1],1))
        target_strength_encodings=self.norm_fm_encoder(target_epsilons_r)
        target_strength_encodings=torch.reshape(target_strength_encodings,[tes[0],tes[1],self.image_size,self.image_size])

        expanded_zero=self.zero_image.expand(target_strength_encodings.shape[0],self.image_channel,self.image_size,self.image_size).cuda()

        #--------------forward rnn sequence-------------------------
        forward_outputs=[]
        for i in range(sn_shape[1]):
            current_inputs=[]
            if i==0:
                current_inputs.append(input_image)
            else:
                current_inputs.append(forward_outputs[i-1])
            current_inputs.append(strength_encodings[:,i:(i+1),:,:])
            current_inputs.append(normalized_noises[:,i,:,:,:])

            input_tensor=torch.cat(current_inputs,1).float()
            output_tensor=self.forward_conv(input_tensor)
            forward_outputs.append(output_tensor)
        forward_outputs.append(expanded_zero)
        forward_outputs=torch.stack(forward_outputs,dim=1)

        #--------------backward rnn sequence-------------------------
        backward_outputs=[]
        for i in range(sn_shape[1],0,-1):
            current_inputs=[]
            if i==sn_shape[1]:
                current_inputs.append(input_image)
            else:
                current_inputs.append(backward_outputs[0])
            current_inputs.append(strength_encodings[:,i-1:i,:,:])
            current_inputs.append(normalized_noises[:,i-1,:,:,:])
            input_tensor=torch.cat(current_inputs,1).float()
            output_tensor=self.backward_conv(input_tensor)
            backward_outputs.insert(0,output_tensor)
        backward_outputs.append(expanded_zero)
        backward_outputs=torch.stack(backward_outputs,dim=1)

        noise_norm_np=source_noises_n.cpu().detach().numpy()
        target_norm_np=target_epsilons.cpu().detach().numpy()
        none_index=backward_outputs.shape[1]-1
        index1_matrix=[]
        index2_matrix=[]
        w1_matrix=[]
        w2_matrix=[]
        for j in range(len(noise_norm_np)):
            index1_list=[]
            index2_list=[]
            w1_list=[]
            w2_list=[]
            for i in range(len(target_norm_np[j])):
                index1,index2=self.get_index(noise_norm_np[j],target_norm_np[j][i],none_index)
                index1_list.append(index1)
                index2_list.append(index2)
                if index1!=none_index and index2==none_index:
                    w1=0
                    w2=target_norm_np[j][i]/noise_norm_np[j][index1]
                elif index2!=none_index and index1==none_index:
                    w1=target_norm_np[j][i]/noise_norm_np[j][index2]
                    w2=0
                else:
                    sum=np.abs(target_norm_np[j][i]-noise_norm_np[j][index1])+np.abs(target_norm_np[j][i]-noise_norm_np[j][index2])
                    w1=np.abs(target_norm_np[j][i]-noise_norm_np[j][index1])/sum
                    w2=np.abs(target_norm_np[j][i]-noise_norm_np[j][index2])/sum
                w1_list.append(w1)
                w2_list.append(w2)
            index1_matrix.append(index1_list)
            index2_matrix.append(index2_list)
            w1_matrix.append(w1_list)
            w2_matrix.append(w2_list)

        index1_matrix=torch.tensor(np.asarray(index1_matrix)).type(torch.LongTensor).cuda()
        index2_matrix=torch.tensor(np.asarray(index2_matrix)).type(torch.LongTensor).cuda()
        w1_matrix=torch.tensor(np.asarray(w1_matrix)).float().cuda()
        w2_matrix=torch.tensor(np.asarray(w2_matrix)).float().cuda()

        i1s=index1_matrix.shape
        index1_matrix=torch.reshape(index1_matrix,(i1s[0],i1s[1],1,1,1))
        index1_matrix=index1_matrix.expand(i1s[0],i1s[1],self.image_channel,self.image_size,self.image_size)

        i2s=index2_matrix.shape
        index2_matrix=torch.reshape(index2_matrix,(i2s[0],i2s[1],1,1,1))
        index2_matrix=index2_matrix.expand(i2s[0],i2s[1],self.image_channel,self.image_size,self.image_size)

        w1s=w1_matrix.shape
        w1_matrix=torch.reshape(w1_matrix,(w1s[0],w1s[1],1,1,1))
        w1_matrix=w1_matrix.expand(w1s[0],w1s[1],self.image_channel,self.image_size,self.image_size)

        w2s=w2_matrix.shape
        w2_matrix=torch.reshape(w2_matrix,(w2s[0],w2s[1],1,1,1))
        w2_matrix=w2_matrix.expand(w2s[0],w2s[1],self.image_channel,self.image_size,self.image_size)

        gathered_forward_hidden=torch.gather(forward_outputs,1, index1_matrix)
        gathered_backward_hidden=torch.gather(backward_outputs,1, index2_matrix)

        temp_noises=torch.cat([normalized_noises,torch.unsqueeze(expanded_zero,1)],1)
        gathered_forward_noise=torch.gather(temp_noises,1, index1_matrix)
        w_gathered_forward_noise=w1_matrix*gathered_forward_noise
        gathered_backward_noise=torch.gather(temp_noises,1, index2_matrix)
        w_gathered_backward_noise=w1_matrix*gathered_backward_noise

        delta_d_matrix=w_gathered_forward_noise+w_gathered_backward_noise

        query=torch.cat([gathered_forward_noise,w_gathered_backward_noise,delta_d_matrix],2)
        qs=query.shape
        query=torch.reshape(query,[qs[0]*qs[1],qs[2],qs[3],qs[4]])
        query=torch.squeeze(self.q_conv(query.float()))
        query=torch.reshape(query,[qs[0],qs[1],self.qk_channel])

        key=torch.cat([forward_outputs,backward_outputs],2)
        ks=key.shape
        key=torch.reshape(key,[ks[0]*ks[1],ks[2],ks[3],ks[4]])
        key=self.k_conv(key.float())
        key=torch.reshape(key,[ks[0],ks[1],self.qk_channel])

        value=torch.cat([forward_outputs,backward_outputs],2)
        vs=value.shape
        value=torch.reshape(value,[vs[0]*vs[1],vs[2],vs[3],vs[4]])
        value=self.v_conv(value.float())
        vs2=value.shape
        value=torch.reshape(value,[vs[0],vs[1],vs2[1],vs2[2],vs2[3]])

        attention_map=torch.softmax(torch.matmul(key,torch.transpose(query,1,2)),1)
        attention_map=torch.transpose(attention_map,1,2)

        vs=value.shape
        value_temp=torch.reshape(value,(vs[0],vs[1],vs[2]*vs[3]*vs[4]))
        results=torch.matmul(attention_map,value_temp)
        delta_dd=torch.reshape(results,[vs[0],attention_map.shape[1],vs[2],vs[3],vs[4]])

        result=delta_dd+delta_d_matrix
        result_shape=result.shape
        result_n=torch.norm(torch.reshape(result,[result_shape[0],result_shape[1],result_shape[2]*result_shape[3]*result_shape[4]]),dim=2)
        result_n=torch.reshape(result_n,[result_shape[0],result_shape[1],1,1,1])
        target_norm_t=torch.reshape(target_epsilons,[result_shape[0],result_shape[1],1,1,1])
        result=(result/result_n)*target_norm_t

        return result.float()

def pgd_attack(model, images, labels, loss, eps=0.3, alpha=0.2, iters=7):
    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

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

    output1=(source_noises_combined,input_image_combined,target_epsilons_combined,label_combined,batch_y_combined)
    output2=(source_noise_list,input_image_list,target_epsilons_list,label_list,order_list,batch_y_list)
    return output1,output2

def renormalize_search_objective(weights,balance):
    new_weights=[]
    new_weights.append(weights[0]*balance[0])
    subweight=[]
    subw_sum=0
    for i in range(len(weights[1])):
        subw_sum+=weights[1][i]
    for i in range(len(weights[1])):
        subweight.append((weights[1][i]/subw_sum)*balance[1])
    new_weights.append(subweight)
    return new_weights

def main():
    AE_M_size=600
    pgd_alpha=0.03
    pgd_step=7
    target_epsilons=[2.3,3.9,5.5]
    extra_strengths=[0.17,0.25]
    full_ae_strength_list=[0.03,0.045,0.06,0.075,0.09,0.105]
    ae_batch_per_nas_step=10
    image_size=32
    image_channel=3
    search_objective_weights=[1,[1,2,3,5,7,10]]
    search_objective_balance=[0.8,0.2]
    search_objective_weights=renormalize_search_objective(search_objective_weights,search_objective_balance)


    warmup_break_step=None
    warmup_eps=0.03
    warmup_alpha=0.03*(2.5/7)
    warmup_iters=7

    s_break_step=None
    s_eps=warmup_eps
    s_alpha=warmup_alpha
    s_iters=warmup_iters


    bm_iters=7
    s_pgd_step=7

    val_break_step=None

    batch_per_model=ae_batch_per_nas_step


    noise_estimator=AN_estimator(image_size,image_channel)
    noise_estimator=noise_estimator.cuda()

    optimizer_ne = torch.optim.AdamW(noise_estimator.parameters(), lr=0.0005, weight_decay=0.0001)
    lambda1=lambda epoch:(epoch/4000) if epoch<4000 else 0.5*(math.cos((epoch-4000)/(100*1000-4000)*math.pi)+1)
    scheduler_ne=optim.lr_scheduler.LambdaLR(optimizer_ne,lr_lambda=lambda1)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # enable GPU and set random seeds
    np.random.seed(args.seed)                  # set random seed: numpy
    torch.cuda.set_device(args.gpu)

    cudnn.deterministic = False
    cudnn.benchmark = True

    torch.manual_seed(args.seed)               # set random seed: torch
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)          # set random seed: torch.cuda
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    if len(unknown_args) > 0:
        logging.warning('unknown_args: %s' % unknown_args)
    else:
        logging.info('unknown_args: %s' % unknown_args)
    # use cross entropy as loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to('cuda')

    # build the model with model_search.Network
    logging.info("init arch param")
    model = Network(C=args.init_channels, num_classes=CIFAR_CLASSES,
                    layers=args.layers, criterion=criterion, tau=args.tau)
    model = model.to('cuda')
    logging.info("model param size = %fMB", utils.count_parameters_in_MB(model))
    log_genotype(model)

    diw = DimensionImportanceWeight(model=model, v_type='mean') if args.diw else None

    # use SGD to optimize the model (optimize model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # generate data indices
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # split training set and validation queue given indices
    # train queue:
    train_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.num_workers
    )

    # validation queue:
    test_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.num_workers
    )

    # learning rate scheduler (with cosine annealing)
    scheduler = CosineAnnealingLR(optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    # construct architect with architect.Architect
    _, feature_num = torch.cat(model.arch_parameters()).shape
    if args.predictor_type == 'lstm':
        is_gae = False
        # -- preprocessor --
        preprocessor = None
        # -- build model --
        predictor = Predictor(input_size=feature_num, hidden_size=args.predictor_hidden_state)
        predictor = predictor.to('cuda')
        reconstruct_criterion = None
    elif args.predictor_type == 'gae':
        is_gae = True
        # -- preprocessor --
        preprocessor = GraphPreprocessor(mode=args.preprocess_mode, lamb=args.preprocess_lamb)
        # -- build model --
        predictor = Estimator(
            extractor=GAEExtractor(
                encoder=GINEncoder(
                    input_dim=args.opt_num, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                    num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers
                ),
                decoder=LinearDecoder(
                    latent_dim=args.latent_dim, decode_dim=args.opt_num, dropout=args.dropout,
                    activation_adj=torch.sigmoid, activation_opt=torch.softmax
                )
            ),
            predictor=PredictorForGraph(in_features=args.latent_dim * 2, out_features=1)
        )
        predictor = predictor.to('cuda')
        reconstruct_criterion = ReconstructedLoss(
            loss_opt=torch.nn.BCELoss(), loss_adj=F.mse_loss, w_opt=1.0, w_adj=1.0
        )
    else:
        raise ValueError('unknown estimator type: %s' % args.predictor_type)

    if args.weighted_loss:
        logging.info('using weighted MSE loss for predictor')
        predictor_criterion = weighted_loss
    else:
        logging.info('using MSE loss for predictor')
        predictor_criterion = F.mse_loss

    architect = Architect(search_objective_weights=search_objective_weights,
        model=model, momentum=args.momentum, weight_decay=args.weight_decay,
        arch_learning_rate=args.arch_learning_rate, arch_weight_decay=args.arch_weight_decay,
        predictor=predictor, pred_learning_rate=args.pred_learning_rate,
        architecture_criterion=F.mse_loss, predictor_criterion=predictor_criterion,
        is_gae=is_gae, reconstruct_criterion=reconstruct_criterion, preprocessor=preprocessor
    )

    if args.evolution:
        memory = Population(batch_size=args.predictor_batch_size, tau=args.tau, is_gae=is_gae)
    else:
        memory = Memory(limit=args.memory_size, batch_size=args.predictor_batch_size, is_gae=is_gae)

    AE_M = AE_Memory(limit=AE_M_size, batch_size=args.predictor_batch_size)

    # --- Part 1: model warm-up and build memory---
    # 1.1 model warm-up
    if args.load_model is not None:
        # load from file
        logging.info('Load warm-up from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_model, 'model-weights-warm-up.pt')))
        warm_up_gumbel = utils.pickle_load(os.path.join(args.load_model, 'gumbel-warm-up.pickle'))
    else:
        # 1.1.1 sample cells for warm-up
        warm_up_gumbel = []
        # assert args.warm_up_population >= args.predictor_batch_size
        for epoch in range(args.warm_up_population):
            g_normal = gumbel_like(model.alphas_normal)
            g_reduce = gumbel_like(model.alphas_reduce)
            warm_up_gumbel.append((g_normal, g_reduce))
        utils.pickle_save(warm_up_gumbel, os.path.join(args.save, 'gumbel-warm-up.pickle'))
        # 1.1.2 warm up
        for epoch, gumbel in enumerate(warm_up_gumbel):
            logging.info('[warm-up model] epoch %d/%d', epoch + 1, args.warm_up_population)
            # warm-up
            model.g_normal, model.g_reduce = gumbel
            #model_train(train_queue, model, criterion, optimizer, name,break_step=None,pgd_epsilon=None,pgd_alpha=None,pgd_step=None)
            objs, top1, top5 = model_train(train_queue=train_queue,
                                            model=model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            name='warm-up model',
                                            break_step=warmup_break_step,
                                            pgd_epsilon=warmup_eps,
                                            pgd_alpha=warmup_alpha,
                                            pgd_step=warmup_iters)
            logging.info('[warm-up model] epoch %d/%d overall loss=%.4f top1-acc=%.4f top5-acc=%.4f',
                         epoch + 1, args.warm_up_population, objs, top1, top5)
            # save weights
            utils.save(model, os.path.join(args.save, 'model-weights-warm-up.pt'))
            # gpu info
            gpu_usage()


    # 1.2 build memory (i.e. valid model)
    if args.load_memory is not None:
        logging.info('Load valid model from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_memory, 'model-weights-valid.pt')))
        memory.load_state_dict(
            utils.pickle_load(
                os.path.join(args.load_memory, 'memory-warm-up.pickle')
            )
        )
    else:
        #---------warmup AE memory and memory -------
        for epoch, gumbel in enumerate(warm_up_gumbel):
            # re-sample Gumbel distribution
            model.g_normal, model.g_reduce = gumbel
            # train model for one step
            objs, top1, top5 = model_train(train_queue=train_queue,
                                            model=model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            name='build memory',
                                            break_step=warmup_break_step,
                                            pgd_epsilon=warmup_eps,
                                            pgd_alpha=warmup_alpha,
                                            pgd_step=warmup_iters)
            logging.info('[build AE memory] train model-%03d loss=%.4f top1-acc=%.4f',
                         epoch + 1, objs, top1)
            #objs, top1, top5 = model_valid(test_queue, model, criterion, name='build memory',val_break_step)
            objs_clean, top1_clean, top5_clean=model_valid(valid_queue=test_queue, model=model, criterion=criterion, name='build memory (clean accuracy)',val_break_step=warmup_break_step)

            utils.save(model, os.path.join(args.save, 'model-weights-valid.pt'))
            utils.pickle_save(memory.state_dict(),
                              os.path.join(args.save, 'memory-warm-up.pickle'))

            ae_loss_list=[]
            for i in range(6):
                ae_loss_list.append(0)
            count=0

            for step, (batch_x, batch_y) in enumerate(train_queue):
                if step>=batch_per_model or batch_x.shape[0]!=args.batch_size:
                    break
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
                images0,noises0=pgd_attack(model, batch_x, batch_y, criterion, eps=0.03, alpha=0.03*(2.5/bm_iters), iters=bm_iters)
                images1,noises1=pgd_attack(model, batch_x, batch_y, criterion, eps=0.06, alpha=0.06*(2.5/bm_iters), iters=bm_iters)
                images2,noises2=pgd_attack(model, batch_x, batch_y, criterion, eps=0.09, alpha=0.09*(2.5/bm_iters), iters=bm_iters)
                temp_eps=np.random.rand()*0.15
                images3,noises3=pgd_attack(model, batch_x, batch_y, criterion, eps=temp_eps, alpha=temp_eps*(2.5/bm_iters), iters=bm_iters)

                output1,output2=form_data(noises0,noises1,noises2,noises3,batch_x,image_channel,image_size,args.batch_size,batch_y)

                source_noise_combined,input_image_combined,target_epsilons_combined,label_combined,batch_yv_combined=output1


                s_noises=torch.stack([noises0,noises1,noises2],1)
                target_epsilons_=torch.unsqueeze(torch.tensor(np.asarray(target_epsilons)),0).cuda()
                target_epsilons_=target_epsilons_.expand(noises0.shape[0],len(target_epsilons))
                temp_results=noise_estimator(source_noises=s_noises,input_image=batch_x,target_epsilons=target_epsilons_)

                ae_list=[]
                ae_list.append(images0)
                ae_list.append(temp_results[:,0,:,:,:]+batch_x)
                ae_list.append(images1)
                ae_list.append(temp_results[:,1,:,:,:]+batch_x)
                ae_list.append(images2)
                ae_list.append(temp_results[:,2,:,:,:]+batch_x)

                for i in range(len(source_noise_combined)):
                    AE_M.append(input_image_combined[i],source_noise_combined[i],target_epsilons_combined[i],label_combined[i],batch_yv_combined[i])


            for step, (batch_x, batch_y) in enumerate(test_queue):
                if step==ae_val_break or batch_x.shape[0]!=args.batch_size:
                    break
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()
                images0,noises0=pgd_attack(model, batch_x, batch_y, criterion, eps=0.03, alpha=0.03*(2.5/bm_iters), iters=bm_iters)
                images1,noises1=pgd_attack(model, batch_x, batch_y, criterion, eps=0.06, alpha=0.06*(2.5/bm_iters), iters=bm_iters)
                images2,noises2=pgd_attack(model, batch_x, batch_y, criterion, eps=0.09, alpha=0.09*(2.5/bm_iters), iters=bm_iters)
                temp_eps=np.random.rand()*0.15
                images3,noises3=pgd_attack(model, batch_x, batch_y, criterion, eps=temp_eps, alpha=temp_eps*(2.5/bm_iters), iters=bm_iters)

                output1,output2=form_data(noises0,noises1,noises2,noises3,batch_x,image_channel,image_size,args.batch_size,batch_y)

                source_noise_combined,input_image_combined,target_epsilons_combined,label_combined,batch_yv_combined=output1


                s_noises=torch.stack([noises0,noises1,noises2],1)
                target_epsilons_=torch.unsqueeze(torch.tensor(np.asarray(target_epsilons)),0).cuda()
                target_epsilons_=target_epsilons_.expand(noises0.shape[0],len(target_epsilons))
                temp_results=noise_estimator(source_noises=s_noises,input_image=batch_x,target_epsilons=target_epsilons_)
                temp_results=temp_results.float()

                ae_list=[]
                ae_list.append(images0)
                ae_list.append(temp_results[:,0,:,:,:]+batch_x)
                ae_list.append(images1)
                ae_list.append(temp_results[:,1,:,:,:]+batch_x)
                ae_list.append(images2)
                ae_list.append(temp_results[:,2,:,:,:]+batch_x)

                for i in range(len(ae_list)):
                    predictions=model(ae_list[i])
                    loss = criterion(predictions, batch_y)
                    loss_ = loss.cpu().detach().numpy()
                    ae_loss_list[i]+=loss_

                count+=images0.shape[0]

            memory.append(weights=[w.detach() for w in model.arch_weights(cat=False)],
                            loss=torch.tensor(objs_clean, dtype=torch.float32).to('cuda'),
                            ae_loss0=torch.tensor(ae_loss_list[0]/count, dtype=torch.float32).to('cuda'),
                            ae_loss1=torch.tensor(ae_loss_list[1]/count, dtype=torch.float32).to('cuda'),
                            ae_loss2=torch.tensor(ae_loss_list[2]/count, dtype=torch.float32).to('cuda'),
                            ae_loss3=torch.tensor(ae_loss_list[3]/count, dtype=torch.float32).to('cuda'),
                            ae_loss4=torch.tensor(ae_loss_list[4]/count, dtype=torch.float32).to('cuda'),
                            ae_loss5=torch.tensor(ae_loss_list[5]/count, dtype=torch.float32).to('cuda'))

            #-------warmup AE estimator with AE_M----------
            AE_estimator_train(image_channel,image_size,criterion,pgd_alpha,pgd_step,model,noise_estimator,AE_M,optimizer_ne,test_queue,scheduler_ne)
    logging.info('memory size=%d', len(memory))

    # --- Part 2 predictor warm-up ---
    if args.load_extractor is not None:
        logging.info('Load extractor from %s', args.load_extractor)
        architect.predictor.extractor.load_state_dict(torch.load(args.load_extractor)['weights'])

    architect.predictor.train()
    for epoch in range(args.predictor_warm_up):
        epoch += 1
        # warm-up
        p_loss, p_true, p_pred = predictor_train(architect=architect, memory=memory)
        if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
            utils.save(architect.predictor, os.path.join(args.save, 'predictor-warm-up.pt'))
    p_loss, p_true, p_pred = predictor_train(architect=architect,memory=memory,print_log=True)

    # gpu info
    gpu_usage()
    # log genotype
    log_genotype(model)

    # --- Part 3 architecture search ---
    for epoch in range(args.epochs):
        # get current learning rate
        lr = scheduler.get_lr()[0]
        logging.info('[architecture search] epoch %d/%d lr %e', epoch + 1, args.epochs, lr)
        # search
        objs, top1, top5, objp = architecture_search(extra_strengths,val_break_step,s_break_step,s_eps,s_alpha,s_iters,
                                                    scheduler_ne,target_epsilons,image_channel,image_size,
                                                    full_ae_strength_list,pgd_alpha,
                                                    s_pgd_step,train_queue, test_queue, model,noise_estimator, architect,
                                                     criterion, optimizer,optimizer_ne, memory,AE_M, ae_batch_per_nas_step, diw)
        # save weights
        utils.save(model, os.path.join(args.save, 'model-weights-search.pt'))
        # log genotype
        log_genotype(model)
        # update learning rate
        scheduler.step()
        # log
        logging.info('[architecture search] overall loss=%.4f top1-acc=%.4f top5-acc=%.4f predictor_loss=%.4f',
                     objs, top1, top5, objp)
        # gpu info
        gpu_usage()


def log_genotype(model):
    # log genotype (i.e. alpha)
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    logging.info('alphas_normal: %s\n%s', torch.argmax(model.alphas_normal, dim=-1), model.alphas_normal)
    logging.info('alphas_reduce: %s\n%s', torch.argmax(model.alphas_reduce, dim=-1), model.alphas_reduce)


def model_train(train_queue, model, criterion, optimizer, name,break_step=None,pgd_epsilon=None,pgd_alpha=None,pgd_step=None):
    # set model to training model
    model.train()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # training loop
    total_steps = len(train_queue)
    for step, (x, target) in enumerate(train_queue):
        optimizer.zero_grad()
        if step==break_step:
            break
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda')
        target = target.to('cuda', non_blocking=True).requires_grad_(False)

        x=x.cuda()
        target=target.cuda()


        x,_ = pgd_attack(model, x, target, criterion,pgd_epsilon,pgd_epsilon*(2.5/pgd_step),pgd_step)


        x = x.requires_grad_(False)
        #x = x.to('cuda').requires_grad_(False)

        logits = model(x)
        loss = criterion(logits, target)


        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('[%s] train model %03d/%03d loss=%.4f top1-acc=%.4f top5-acc=%.4f',
                         name, step, total_steps, objs.avg, top1.avg, top5.avg)
    # return average metrics
    return objs.avg, top1.avg, top5.avg


def model_valid(valid_queue, model, criterion, name,val_break_step=None):
    # set model to evaluation model
    model.eval()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # validation loop
    total_steps = len(valid_queue)
    for step, (x, target) in enumerate(valid_queue):
        if step==val_break_step:
            break

        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # valid model
        logits = model(x)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        # log
        if step % args.report_freq == 0:
            logging.info('[%s] valid model %03d/%03d loss=%.4f top1-acc=%.4f top5-acc=%.4f',
                         name, step, total_steps, objs.avg, top1.avg, top5.avg)
    return objs.avg, top1.avg, top5.avg

def model_valid_ae(valid_queue, model, criterion, name, ae_strength_list,val_break_step=None,extra_strengths=None,pgd_iters=20):

    # set model to evaluation model
    model.eval()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # validation loop
    total_steps = len(valid_queue)

    top1_list=[]

    for i in range(len(ae_strength_list)):
        top1 = utils.AverageMeter()
        for step, (x, target) in enumerate(valid_queue):
            if step==val_break_step:
                break

            n = x.size(0)
            # data to CUDA
            x = x.to('cuda').requires_grad_(False)
            target = target.to('cuda', non_blocking=True).requires_grad_(False)
            x,_=pgd_attack(model=model, images=x, labels=target, loss=criterion, eps=ae_strength_list[i], alpha=ae_strength_list[i]*(2.5/pgd_iters), iters=pgd_iters)
            # valid model
            logits = model(x)
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            # update metrics
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)

        top1_list.append(top1.avg)
    if extra_strengths==None:
        logging.info('valid model on AE [---AE val acc---]: eps0:%.4f, eps1:%.4f, eps2:%.4f, eps3:%.4f, eps4:%.4f, eps5:%.4f',
                    top1_list[0],top1_list[1],top1_list[2],top1_list[3],top1_list[4],top1_list[5])
    else:
        top1_list_extra=[]
        for i in range(len(extra_strengths)):
            top1 = utils.AverageMeter()
            for step, (x, target) in enumerate(valid_queue):
                if step==val_break_step:
                    break

                n = x.size(0)
                # data to CUDA
                x = x.to('cuda').requires_grad_(False)
                target = target.to('cuda', non_blocking=True).requires_grad_(False)
                x,_=pgd_attack(model=model, images=x, labels=target, loss=criterion, eps=extra_strengths[i], alpha=extra_strengths[i]*(2.5/pgd_iters), iters=pgd_iters)
                # valid model
                logits = model(x)
                loss = criterion(logits, target)
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                # update metrics
                objs.update(loss.data.item(), n)
                top1.update(prec1.data.item(), n)
            top1_list_extra.append(top1.avg)
        logging.info('valid model on AE [---AE val acc---]: eps0:%.4f, eps1:%.4f, eps2:%.4f, eps3:%.4f, eps4:%.4f, eps5:%.4f',
                    top1_list[0],top1_list[1],top1_list[2],top1_list[3],top1_list[4],top1_list[5])
        log_string='valid model on AE (extra):'
        logging.info('valid model on AE (extra):')
        for i in range(len(extra_strengths)):
            log_string='extra-eps'+str(i)+':%.4f,'
            logging.info(log_string,
                        top1_list_extra[i])


def AE_estimator_train(image_channel,image_size,criterion,pgd_alpha,pgd_step,model,noise_estimator,AE_M,optimizer_ne,val_loader,scheduler_ne):
    model.train()
    batch = AE_M.get_batch()
    noise_estimator.train()
    counter=0
    for input_image_list,source_noise_list,target_epsilons_list,label_list,batch_y_list in batch:
        counter+=1
        optimizer_ne.zero_grad()

        n_estimate=noise_estimator(source_noise_list,input_image_list,target_epsilons_list)
        temp=torch.squeeze(n_estimate,1)-label_list
        loss=temp*temp
        loss=torch.mean(loss)
        loss.backward()
        optimizer_ne.step()
        scheduler_ne.step()


    torch.save(noise_estimator.state_dict(), current_folder+'/'+model_name+'/'+model_name+'_'+est_name)

def predictor_train(architect, memory, unsupervised=False,print_log=False):
    # TODO: add support for gae predictor training
    objs = utils.AverageMeter()
    batch = memory.get_batch()
    all_y = []
    all_p = []

    loss_ori_am = utils.AverageMeter()
    loss_0_am = utils.AverageMeter()
    loss_1_am = utils.AverageMeter()
    loss_2_am = utils.AverageMeter()
    loss_3_am = utils.AverageMeter()
    loss_4_am = utils.AverageMeter()
    loss_5_am = utils.AverageMeter()
    loss_sum_am = utils.AverageMeter()


    for x,y,ae_loss0,ae_loss1,ae_loss2,ae_loss3,ae_loss4,ae_loss5 in batch:
        #print('here',x.shape,y.shape)
        n = x.size(0)
        pred, loss = architect.predictor_step(x,y,ae_loss0,ae_loss1,ae_loss2,ae_loss3,ae_loss4,ae_loss5, unsupervised=unsupervised)
        objs.update(loss.data.item(), n)
        all_y.append(y)
        all_p.append(pred)

        out,oute0,oute1,oute2,oute3,oute4,oute5=architect.predictor(x)
        # calculate loss
        loss_ori = torch.nn.functional.mse_loss(out, y)
        loss_0 = torch.nn.functional.mse_loss(oute0, ae_loss0)
        loss_1 = torch.nn.functional.mse_loss(oute1, ae_loss1)
        loss_2 = torch.nn.functional.mse_loss(oute2, ae_loss2)
        loss_3 = torch.nn.functional.mse_loss(oute3, ae_loss3)
        loss_4 = torch.nn.functional.mse_loss(oute4, ae_loss4)
        loss_5 = torch.nn.functional.mse_loss(oute5, ae_loss5)
        loss=loss_ori+loss_0+loss_1+loss_2+loss_3+loss_4+loss_5


        loss_ori_am.update(loss_ori.data.item(), n)
        loss_0_am.update(loss_0.data.item(), n)
        loss_1_am.update(loss_1.data.item(), n)
        loss_2_am.update(loss_2.data.item(), n)
        loss_3_am.update(loss_3.data.item(), n)
        loss_4_am.update(loss_4.data.item(), n)
        loss_5_am.update(loss_5.data.item(), n)
        loss_sum_am.update(loss.data.item(), n)
    if print_log==True:
        logging.info('AR estimator performance: \n overall loss: %.4f \n loss ori: %.4f \n loss0: %.4f \n loss1: %.4f\n loss2: %.4f\n loss3: %.4f\n loss4: %.4f\n loss5: %.4f\n',
                    loss_sum_am.avg,loss_ori_am.avg,loss_0_am.avg,loss_1_am.avg,loss_2_am.avg,loss_3_am.avg,loss_4_am.avg,loss_5_am.avg)

    return objs.avg, torch.cat(all_y), torch.cat(all_p)

def architecture_search(extra_strengths,val_break_step,s_break_step,s_eps,s_alpha,s_iters,
                        scheduler_ne,target_epsilons,image_channel,image_size,
                        full_ae_strength_list,pgd_alpha,pgd_step,train_queue, valid_queue, model,noise_estimator,
                        architect, criterion, optimizer,optimizer_ne, memory,AE_M,
                        ae_batch_per_nas_step=None,diw: Optional[DimensionImportanceWeight]=None):
    # -- train model --
    if diw is not None and diw.num > 0:
        diw_normal, diw_reduce = diw.get_diw()  # diw weight
        gsw_normal = 1. - diw_normal  # gumbel sampling weight
        gsw_reduce = 1. - diw_reduce  # gumbel sampling weight
    else:
        gsw_normal, gsw_reduce = 1., 1.  # gumbel sampling weight
    model.g_normal = gumbel_like(model.alphas_normal) * gsw_normal
    model.g_reduce = gumbel_like(model.alphas_reduce) * gsw_reduce


    # train model for one step
    objs, top1, top5 = model_train(train_queue=train_queue,
                                    model=model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    name='NAS - train model',
                                    break_step=s_break_step,
                                    pgd_epsilon=s_eps,
                                    pgd_alpha=s_alpha,
                                    pgd_step=s_iters)

    # update AE_M
    for step, (batch_x, batch_y) in enumerate(train_queue):
        #print('step',step)
        if step>ae_batch_per_nas_step or batch_x.shape[0]!=args.batch_size:
            break

        batch_x=batch_x.cuda()
        batch_y=batch_y.cuda()

        images0,noises0=pgd_attack(model, batch_x, batch_y, criterion, eps=0.03, alpha=0.03*(2.5/7), iters=7)
        images1,noises1=pgd_attack(model, batch_x, batch_y, criterion, eps=0.06, alpha=0.06*(2.5/7), iters=7)
        images2,noises2=pgd_attack(model, batch_x, batch_y, criterion, eps=0.09, alpha=0.09*(2.5/7), iters=7)
        temp_eps=np.random.rand()*0.15
        images3,noises3=pgd_attack(model, batch_x, batch_y, criterion, eps=temp_eps, alpha=temp_eps*(2.5/7), iters=7)

        output1,output2=form_data(noises0,noises1,noises2,noises3,batch_x,image_channel,image_size,args.batch_size,batch_y)
        source_noise_combined,input_image_combined,target_epsilons_combined,label_combined,batch_yv_combined=output1

        s_noises=torch.stack([noises0,noises1,noises2],1)
        target_epsilons_=torch.unsqueeze(torch.tensor(np.asarray(target_epsilons)),0).cuda()
        target_epsilons_=target_epsilons_.expand(noises0.shape[0],len(target_epsilons))
        temp_results=noise_estimator(source_noises=s_noises,input_image=batch_x,target_epsilons=target_epsilons_)

        ae_list=[]
        ae_list.append(images0)
        ae_list.append(temp_results[:,0,:,:,:]+batch_x)
        ae_list.append(images1)
        ae_list.append(temp_results[:,1,:,:,:]+batch_x)
        ae_list.append(images2)
        ae_list.append(temp_results[:,2,:,:,:]+batch_x)


        for i in range(len(source_noise_combined)):
            AE_M.append(input_image_combined[i],source_noise_combined[i],target_epsilons_combined[i],label_combined[i],batch_yv_combined[i])

    ae_top1=[]
    for i in range(6):
        top1 = utils.AverageMeter()
        ae_top1.append(top1)

    ae_loss_list=[]
    for i in range(6):
        ae_loss_list.append(0)
    count=0

    for step, (batch_x, batch_y) in enumerate(valid_queue):
        if step==ae_val_break or batch_x.shape[0]!=args.batch_size:
            break

        batch_x=batch_x.cuda()
        batch_y=batch_y.cuda()

        images0,noises0=pgd_attack(model, batch_x, batch_y, criterion, eps=0.03, alpha=0.03*(2.5/7), iters=7)
        images1,noises1=pgd_attack(model, batch_x, batch_y, criterion, eps=0.06, alpha=0.06*(2.5/7), iters=7)
        images2,noises2=pgd_attack(model, batch_x, batch_y, criterion, eps=0.09, alpha=0.09*(2.5/7), iters=7)
        temp_eps=np.random.rand()*0.15
        images3,noises3=pgd_attack(model, batch_x, batch_y, criterion, eps=temp_eps, alpha=temp_eps*(2.5/7), iters=7)

        output1,output2=form_data(noises0,noises1,noises2,noises3,batch_x,image_channel,image_size,args.batch_size,batch_y)
        source_noise_combined,input_image_combined,target_epsilons_combined,label_combined,batch_yv_combined=output1

        s_noises=torch.stack([noises0,noises1,noises2],1)
        target_epsilons_=torch.unsqueeze(torch.tensor(np.asarray(target_epsilons)),0).cuda()
        target_epsilons_=target_epsilons_.expand(noises0.shape[0],len(target_epsilons))
        temp_results=noise_estimator(source_noises=s_noises,input_image=batch_x,target_epsilons=target_epsilons_)

        ae_list=[]
        ae_list.append(images0)
        ae_list.append(temp_results[:,0,:,:,:]+batch_x)
        ae_list.append(images1)
        ae_list.append(temp_results[:,1,:,:,:]+batch_x)
        ae_list.append(images2)
        ae_list.append(temp_results[:,2,:,:,:]+batch_x)

        for i in range(len(ae_list)):
            predictions=model(ae_list[i])
            prec1, prec5 = utils.accuracy(predictions, batch_y, topk=(1, 5))
            ae_top1[i].update(prec1.data.item(), images0.shape[0])

        for i in range(len(ae_list)):
            predictions=model(ae_list[i])
            loss = criterion(predictions, batch_y)
            loss_ = loss.cpu().detach().numpy()
            ae_loss_list[i]+=loss_
        count+=images0.shape[0]

    # update AE_estimator
    AE_estimator_train(image_channel,image_size,criterion,pgd_alpha,pgd_step,model,noise_estimator,AE_M,optimizer_ne,valid_queue,scheduler_ne)
    logging.info('valid model on AE [---AE val acc---]: eps0:%.4f, eps2:%.4f, eps4:%.4f',
            ae_top1[0].avg,ae_top1[2].avg,ae_top1[4].avg)
    # -- valid model --
    objs, top1, top5 = model_valid(valid_queue=train_queue, model=model, criterion=criterion, name='NAS - model clean validate',val_break_step=val_break_step)
    # save validation to memory
    logging.info('[architecture search] append memory [---clean train acc---] objs=%.4f top1-acc=%.4f top5-acc=%.4f', objs, top1, top5)

    objs_val, top1_val, top5_val = model_valid(valid_queue=valid_queue, model=model, criterion=criterion, name='NAS - model clean validate',val_break_step=val_break_step)
    # save validation to memory
    logging.info('[architecture search] append memory [---clean test acc---] objs=%.4f top1-acc=%.4f top5-acc=%.4f', objs_val, top1_val, top5_val)

    if args.evolution:
        memory.append(individual=[(model.alphas_normal.detach().clone(), model.g_normal.detach().clone()),
                                  (model.alphas_reduce.detach().clone(), model.g_reduce.detach().clone())],
                      fitness=torch.tensor(objs, dtype=torch.float32).to('cuda'))
        index = memory.remove('highest')
        logging.info('[evolution] %d is removed (population size: %d).' % (index, len(memory)))
    else:
        memory.append(weights=[w.detach() for w in model.arch_weights(cat=False)],
                        loss=torch.tensor(objs, dtype=torch.float32).to('cuda'),
                        ae_loss0=torch.tensor(ae_loss_list[0]/count, dtype=torch.float32).to('cuda'),
                        ae_loss1=torch.tensor(ae_loss_list[1]/count, dtype=torch.float32).to('cuda'),
                        ae_loss2=torch.tensor(ae_loss_list[2]/count, dtype=torch.float32).to('cuda'),
                        ae_loss3=torch.tensor(ae_loss_list[3]/count, dtype=torch.float32).to('cuda'),
                        ae_loss4=torch.tensor(ae_loss_list[4]/count, dtype=torch.float32).to('cuda'),
                        ae_loss5=torch.tensor(ae_loss_list[5]/count, dtype=torch.float32).to('cuda'))

    utils.pickle_save(memory.state_dict(),
                      os.path.join(args.save, 'memory-search.pickle'))

    # -- predictor train --
    architect.predictor.train()

    # use memory to train predictor
    p_loss, p_true, p_pred = None, None, None
    k_tau = -float('inf')
    for i in range(args.predictor_warm_up):
        p_loss, p_true, p_pred = predictor_train(architect=architect,memory= memory)
    p_loss, p_true, p_pred = predictor_train(architect=architect,memory=memory,print_log=True)
    # -- architecture update --
    if args.evolution:
        index, weights, _ = memory.select('lowest')
        logging.info('[evolution] %d is selected (population size: %d).' % (index, len(memory)))
        (a_normal, g_normal), (a_reduce, g_reduce) = weights
        model.alphas_normal.data = a_normal
        model.alphas_reduce.data = a_reduce
        model.g_normal = g_normal
        model.g_reduce = g_reduce
    architect.step()
    # log
    logging.info('[architecture search] update architecture')

    if diw is not None:
        diw.update()
        diw_normal, diw_reduce = diw.get_diw()
        logging.info('diw_normal: %s' % (diw_normal,))
        logging.info('diw_reduce: %s' % (diw_reduce,))

    logging.info('-----------------------------------------')

    return objs, top1, top5, p_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    # data
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--num_workers', type=int, default=2, help='number of data loader workers')#4 in the original code
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    # save
    parser.add_argument('--save', type=str, default='EXP'+model_name, help='experiment name')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    # training setting
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate') #0.1 as in the TRADES original paper
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    # search setting
    parser.add_argument('--arch_learning_rate', type=float, default=3e-5, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--memory_size', type=int, default=100, help='size of memory to train predictor')
    parser.add_argument('--warm_up_population', type=int, default=100, help='warm_up_population')
    parser.add_argument('--load_model', type=str, default=None, help='load model weights from file')
    parser.add_argument('--load_memory', type=str, default=None, help='load memory from file')
    parser.add_argument('--tau', type=float, default=0.1, help='tau')
    parser.add_argument('--evolution', action='store_true', default=False, help='use weighted loss')
    parser.add_argument('--diw', action='store_true', default=False, help='dimension importance aware')
    # predictor setting
    parser.add_argument('--predictor_type', type=str, default='lstm')
    parser.add_argument('--predictor_warm_up', type=int, default=500, help='predictor warm-up steps')
    parser.add_argument('--predictor_hidden_state', type=int, default=16, help='predictor hidden state')
    parser.add_argument('--predictor_batch_size', type=int, default=64, help='predictor batch size')
    parser.add_argument('--pred_learning_rate', type=float, default=1e-3, help='predictor learning rate')
    parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
    parser.add_argument('--load_extractor', type=str, default=None, help='load memory from file')
    # model setting
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    # others
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--debug', action='store_true', default=False, help='set logging level to debug')
    # GAE related
    parser.add_argument('--opt_num', type=int, default=11)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    # parser.add_argument('--threshold', type=float, default=0.5)  # TODO: unused arg
    # optimization
    # parser.add_argument('--lr', type=float, default=1e-3)  # TODO: unused arg
    # parser.add_argument('--eps', type=float, default=1e-8)  # TODO: unused arg
    # parser.add_argument('--supervised', action='store_true', default=False)  # TODO: unused arg
    # data
    parser.add_argument('--preprocess_mode', type=int, default=4)
    parser.add_argument('--preprocess_lamb', type=float, default=0.)

    args, unknown_args = parser.parse_known_args()

    args.save = 'checkpoints/search{}-{}-{}'.format(
        '-ea' if args.evolution else '', args.save, time.strftime("%Y%m%d-%H%M%S")
    )
    utils.create_exp_dir(
        path=args.save,
        scripts_to_save=glob.glob('*.py') + glob.glob('module/**/*.py', recursive=True)
    )

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging_level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(stream=sys.stdout, level=logging_level,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(model_name)

    if not os.path.exists(current_folder+'/'+model_name):
        os.mkdir(current_folder+'/'+model_name)

    main()
