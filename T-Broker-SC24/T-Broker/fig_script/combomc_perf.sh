#!/bin/bash
cat $1 | grep END | wc -l
A=$(cat $1 | grep "\[JOB\]" | awk -v FS='TIMESTAMP' '{print $NF}'|grep BEGIN | awk -v FS=',' '{print $NF}')
B=`cat $1 | grep "\[JOB\]" | awk -v FS='TIMESTAMP' '{print $NF}'|grep END | awk -v FS=',' '{print $NF}'`
#T=`cat $1 | grep TcpAgent | tr '>' "'" | tr '<' "'" | python3 ready.py`
#echo $A $B | python3 perf.py $T
echo $A $B | python3 perf.py 
#cat $1 | grep "\[JOB\]" | awk -v FS='TIMESTAMP' '{print $NF}'|grep BEGIN | awk -v FS=',' '{print $NF}'
