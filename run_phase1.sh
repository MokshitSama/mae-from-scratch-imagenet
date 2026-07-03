#!/usr/bin/env bash
# Phase 1 - accelerate DDP on the six 5090s (PCI 0,1,2,3,5,6; index 4 is the 3090, EXCLUDED).
# NOTE: use --gpu_ids, NOT CUDA_VISIBLE_DEVICES — accelerate ignores the latter and would
#       put a rank on the 3090 (GPU 4) -> illegal memory access.
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
cd "$(dirname "$0")/phase1"
accelerate launch --multi_gpu --num_processes 6 --gpu_ids 0,1,2,3,5,6 --main_process_port 29501 train.py
