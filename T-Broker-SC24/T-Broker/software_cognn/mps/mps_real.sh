#!/bin/bash
NOW=`python3 now.py`
export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software_cognn:${MIMOSE_PATH}/src"
    while true
    do
	    break
        process=`ps aux | grep run_real | grep -v grep`
        if [ "$process" == "" ]; then
            break
        fi
    done
for i in `seq 0 35`;
do
    while true
    do
        process=`ps aux | grep mps_main | grep -v grep`
        if [ "$process" == "" ]; then
            taskId=`expr $i \* 2`    
            CUDA_VISIBLE_DEVICES=0 python mps_main.py $1 ${taskId} $NOW &
            taskId=`expr ${taskId} + 1`
            CUDA_VISIBLE_DEVICES=0 python mps_main.py $1 ${taskId} $NOW
            break
        fi
    done
done
