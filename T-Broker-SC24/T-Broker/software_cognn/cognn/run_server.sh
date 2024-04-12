#!/bin/bash
set -u
export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software_cognn:${MIMOSE_PATH}/src"
worker=2
CUDA_VISIBLE_DEVICES=0 python main.py $1 ${worker} $2 $3
