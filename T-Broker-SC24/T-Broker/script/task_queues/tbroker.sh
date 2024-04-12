#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0

cd ../../software/cognn
for percent in 20 40 60 80 100;
do
  for policy in 0 1 2;
  do
    echo "start"
    ./run_server.sh ../../data/task_queues/dnn${percent}.txt 11000 ${policy} &> /home/T-Broker-SC24/T-Broker/log/dnn${percent}_${policy}.log
  done
done
cd ../../script/task_queues/
