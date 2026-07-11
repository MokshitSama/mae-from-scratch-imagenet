#!/usr/bin/env bash
# SimMIM pretrain - accelerate DDP on the six 5090s (PCI 0,1,2,3,5,6; index 4 is the 3090, EXCLUDED).
# NOTE: use --gpu_ids, NOT CUDA_VISIBLE_DEVICES — accelerate ignores the latter and would
#       put a rank on the 3090 (GPU 4) -> illegal memory access.
# NCCL: torch's build-time nccl is 2.26.2 (crashes on 5090 all-reduce); pip nvidia-nccl-cu12==2.26.5
#       is installed and torch's RPATH resolves to it; the preload below is belt-and-suspenders.
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1        # else stdout block-buffers to the log file and you see nothing for ages
export V="${1:-${V:-1}}"         # run version from the command line: bash run_simmim.sh 2
NCCL_2265=/home/akane/anaconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
[ -f "$NCCL_2265" ] && export LD_PRELOAD="$NCCL_2265${LD_PRELOAD:+:$LD_PRELOAD}"
cd "$(dirname "$0")/phase2"
accelerate launch --multi_gpu --num_processes 6 --gpu_ids 0,1,2,3,5,6 --main_process_port 29504 train_simmim.py
