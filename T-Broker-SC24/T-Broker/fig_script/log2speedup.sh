for policy in   default mps lucid cognn tbroker0 tbroker1 tbroker2;
	do
		echo $policy
	done
BASE=/home/T-Broker-SC24/T-Broker/log
for i in `seq 1 3`;do
for percent in 20 40 60 80 100;
do
for policy in   default mps lucid cognn  0 1 2;
	do
	DEFAULT=`./combomc_perf.sh ${BASE}/dnn${percent}_default.log | tail -n 3  |head -n $i | tail -n 1 | awk -v FS=':' '{print $2}'`
	EXP=`./combomc_perf.sh ${BASE}/dnn${percent}_${policy}.log | tail -n 3  |head -n $i | tail -n 1 | awk -v FS=':' '{print $2}' `
	SPEEDUP=`python3 div.py $DEFAULT $EXP`
	echo $SPEEDUP
	done
done
done
