DATA=`./log2speedup.sh | paste - - - - - - - -d ","`
HEAD=` echo $DATA | tr ' ' '\n' | head -n 1`
QT=`echo $DATA | tr ' ' '\n' | head -n 6 | tail -n 5`
JCT=`echo $DATA | tr ' ' '\n' | head -n 11 | tail -n 5`
MAKESPAN=`echo $DATA | tr ' ' '\n' | head -n 16 | tail -n 5`
echo $HEAD
echo $QT | tr ' ' '\n'
echo $HEAD
echo $JCT | tr ' ' '\n'
echo $HEAD
echo $MAKESPAN | tr ' ' '\n'
