for policy in   default static dynamic tbroker2;
	do
		echo $policy
	done
BASE=/home/T-Broker-SC24/T-Broker/log
for i in `seq 1 3`;do
for policy in   default DR RC 2;
	do
	DEFAULT=`./combomc_perf.sh ${BASE}/trace_default.log | tail -n 3  |head -n $i | tail -n 1 | awk -v FS=':' '{print $2}'`
	EXP=`./combomc_perf.sh ${BASE}/trace_${policy}.log | tail -n 3  |head -n $i | tail -n 1 | awk -v FS=':' '{print $2}' `
	SPEEDUP=`python3 div.py $DEFAULT $EXP`
	echo $SPEEDUP
done
done
