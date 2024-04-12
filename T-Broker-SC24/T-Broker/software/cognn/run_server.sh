#!/bin/bash
set -u
worker=2
export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software:${MIMOSE_PATH}/src:${MIMOSE_PATH}/examples/pytorch/multiple-choice/:/home/T-Broker-SC24/mimose-mmdet"
#python main.py ../data/model/test_gnn.txt ${worker} $1 $2
python main.py $1 ${worker} $2 $3
