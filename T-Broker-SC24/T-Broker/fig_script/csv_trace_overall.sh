DATA=`./log2speedup_real.sh | paste - - - - - - -d ","`
echo $DATA | tr ' ' '\n'
