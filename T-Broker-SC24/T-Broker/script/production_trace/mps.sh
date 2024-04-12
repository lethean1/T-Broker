#!/bin/bash


export MIMOSE_PATH="/home/T-Broker-SC24/T-Broker/mimose-transformers"
export PYTHONPATH="/home/T-Broker-SC24/T-Broker/software:${MIMOSE_PATH}/src"

cd /home/T-Broker-SC24/T-Broker/software_cognn/mps/ 
./mps_real.sh /home/T-Broker-SC24/T-Broker/data/production_trace/trace.txt &> /home/T-Broker-SC24/T-Broker/log/trace_mps.log

cd ../../script/production_trace/

