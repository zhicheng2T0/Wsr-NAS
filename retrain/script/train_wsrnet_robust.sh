#!/bin/bash
./script/command_mgpu_block.sh t_cna_rp 2 1 "python3 -u train_adv_mgpu_cifar.py --init_channels=45 --layers=20 --save=EXP_cna_r_p --arch=cna_r --num_steps=7 --batch_size=64 --report_freq=250"