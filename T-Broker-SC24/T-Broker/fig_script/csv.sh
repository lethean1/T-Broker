DATA=`./csv_task.sh`
echo $DATA | tr ' ' '\n' | head -n 6 > 1.tmp
paste -d " " col0.txt 1.tmp > task_qt.csv
echo $DATA | tr ' ' '\n' | head -n 12 | tail -n 6 > 1.tmp
paste -d " " col0.txt 1.tmp > task_jct.csv
echo $DATA | tr ' ' '\n' | head -n 18 | tail -n 6 > 1.tmp
paste -d " " col0.txt 1.tmp > task_makespan.csv
DATA=`./csv_ablation.sh`
echo $DATA | tr ' ' '\n' > 1.tmp
paste -d " " col.txt 1.tmp > ablation.csv
DATA=`./csv_trace_overall.sh`
echo $DATA | tr ' ' '\n' > 1.tmp
paste -d " " col.txt 1.tmp > trace_overall.csv
