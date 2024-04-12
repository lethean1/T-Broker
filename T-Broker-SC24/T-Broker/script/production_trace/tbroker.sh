#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0
   
for policy in 0 1 2;
do
	cd /home/T-Broker-SC24/T-Broker/software/cognn/
	./run_server.sh /home/T-Broker-SC24/T-Broker/data/production_trace/trace.txt 10902 ${policy} &> /home/T-Broker-SC24/T-Broker/log/trace_${policy}.log
done
cd ../../script/production_trace/
