#!/bin/bash

set -u

export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software_cognn:${MIMOSE_PATH}/src"
for i in `seq 0 1`;
do
    CUDA_VISIBLE_DEVICES=0 python default_main.py $1 ${i}
done
