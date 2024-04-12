#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0
cd /home/T-Broker-SC24/T-Broker/software/cognn/ 
for policy in lucid;
do
	./run_server.sh /home/T-Broker-SC24/T-Broker/data/production_trace/trace.txt 10902 ${policy} &> /home/T-Broker-SC24/T-Broker/log/trace_${policy}.log
done
cd ../../script/production_trace/
