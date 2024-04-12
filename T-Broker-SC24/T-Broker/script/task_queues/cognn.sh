#!/bin/bash
cd ../../software_cognn/


export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software_cognn:${MIMOSE_PATH}/src"

cd cognn
for percent in 20 40 60 80 100;
do
    python main.py /home/T-Broker-SC24/T-Broker/data/task_queues/dnn${percent}.txt 2 12234 2 &> /home/T-Broker-SC24/T-Broker/log/dnn${percent}_cognn.log
done

cd ../../script/task_queues

