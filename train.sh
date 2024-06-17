#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=8 pretrain.py --train_data_dir data/output
#--resume True
