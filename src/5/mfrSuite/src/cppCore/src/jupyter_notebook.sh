#!/bin/bash

#SBATCH --output=slurm_files/jupyter.out
# module load Anaconda3
ip=$(/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}')
port=$((10000+ $RANDOM % 20000))
# export OPENBLAD_NUM_THREADS=1
jupyter notebook --no-browser --ip=$ip --port=$port --log-level='ERROR'
