#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0

conda activate tbroker
./default.sh
./lucid.sh
./cognn.sh

nvidia-cuda-mps-control -d
./tbroker.sh

conda actiavte mps
./mps.sh

echo quit | nvidia-cuda-mps-control
