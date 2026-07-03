#!/usr/bin/env bash
# Phase 1 (rewrite) - ViT-B/16 from scratch on ImageNet-1K, accelerate DDP on the six 5090s.
# All hyperparameters come from config.py (CFG). Index 4 is the 3090 and is excluded.
#
# NOTE: CFG.epochs is 300 -> this launches the FULL ~25h run.
#       For a 1-epoch speed test, set CFG.epochs = 1 in config.py first.
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6
export OMP_NUM_THREADS=8

accelerate launch --multi_gpu --num_processes 6 --main_process_port 29501 train.py
