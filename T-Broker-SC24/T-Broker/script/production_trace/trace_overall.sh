#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0

conda activate tbroker
./default.sh
./lucid.sh

nvidia-cuda-mps-control -d
./tbroker.sh

conda activate mps
./mps.sh

echo quit | nvidia-cuda-mps-control
