#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0

cd ../../software/cognn/
for percent in 20 40 60 80 100;
do
  ./run_server.sh /home/T-Broker-SC24/T-Broker/data/task_queues/dnn${percent}.txt 11000 lucid &> /home/T-Broker-SC24/T-Broker/log/dnn${percent}_lucid.log
done
cd ../../script/task_queues/
