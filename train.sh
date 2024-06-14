#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=4 pretrain.py --train_data_dir data/output