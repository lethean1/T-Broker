#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0
   
for policy in recompute0_static;
do
	cd /home/T-Broker-SC24/T-Broker/software/cognn/
	./run_server.sh /home/T-Broker-SC24/T-Broker/data/production_trace/trace.txt 10902 ${policy} &> /home/T-Broker-SC24/T-Broker/log/trace_RC.log
done
cd ../../script/production_trace/
