DATA=`./log2speedup_ablation.sh | paste - - - - -d ","`
echo $DATA | tr ' ' '\n'
