#!/bin/bash
cd ../../software_cognn/mps/


export MIMOSE_PATH="/home/T-Broker-SC24/T-Broker/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software_cognn:${MIMOSE_PATH}/src"

for percent in 20 40 60 80 100;
do
    ./run_mps.sh /home/T-Broker-SC24/T-Broker/data/task_queues/dnn${percent}.txt &> /home/T-Broker-SC24/T-Broker/log/dnn${percent}_mps.log
done
cd ../../script/task_queues/

