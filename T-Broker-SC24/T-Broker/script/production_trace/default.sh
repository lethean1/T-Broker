#!/bin/bash


export MIMOSE_PATH="/home/T-Broker-SC24/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software:${MIMOSE_PATH}/src"


cd /home/T-Broker-SC24/T-Broker/software_cognn/default/

./run_default.sh /home/T-Broker-SC24/T-Broker/data/production_trace/trace.txt &> /home/T-Broker-SC24/T-Broker/log/trace_default.log

cd ../../script/production_trace/
