#!/bin/bash
export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software_cognn:${MIMOSE_PATH}/src"
for i in `seq 0 9`;
do
    while true
    do
        process=`ps aux | grep mps_main | grep -v grep`
        if [ "$process" == "" ]; then
            taskId=`expr $i \* 2`    
            CUDA_VISIBLE_DEVICES=0 python mps_main.py $1 ${taskId} &
            taskId=`expr ${taskId} + 1`
            CUDA_VISIBLE_DEVICES=0 python mps_main.py $1 ${taskId}
            break
        fi
    done
done
