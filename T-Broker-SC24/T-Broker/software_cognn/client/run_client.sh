#!/bin/bash
export PYTHONPATH=/home/sqx/CoGNN_info_for_SC22/software
CUDA_VISIBLE_DEVICES=1 python client.py ../data/model/test_20_2.txt
