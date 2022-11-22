#!/bin/bash
#-x BJ-IDC1-10-10-16-[46,83,85,51,53,60,61,86,88] \

job_name=$1
train_gpu=$2
num_node=$3
command=$4
total_process=$((train_gpu*num_node))

mkdir -p log

now=$(date +"%Y%m%d_%H%M%S")

port=$(( $RANDOM % 300 + 23450 ))

# nohup 
GLOG_vmodule=MemcachedClient=-1 \
srun --partition=VA \
--mpi=pmi2 -n$total_process \
--gres=gpu:$train_gpu \
--ntasks-per-node=$train_gpu \
--cpus-per-task=7 \
--job-name=$job_name \
--kill-on-bad-exit=1 \
#--exclude=BJ-IDC1-10-10-16-[46,48,49,51,52,53,60,71,58,61,62,64,66,68,69,84] \
$command --port $port 2>&1|tee -a log/$job_name.log &


