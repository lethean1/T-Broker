#!/bin/bash
set +u
export CUDA_VISIBLE_DEVICES=0

echo quit | nvidia-cuda-mps-control
source activate tbroker

./default.sh

nvidia-cuda-mps-control -d
./RC.sh
./DR.sh
./tbroker.sh

echo quit | nvidia-cuda-mps-control
