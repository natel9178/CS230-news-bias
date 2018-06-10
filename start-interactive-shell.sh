#!/bin/bash
module load cudnn
srun --pty --partition=gpu --gres=gpu:1 --qos=interactive $SHELL -l

#sbatch --partition=gpu --gres=gpu:1 --qos=gpu --time=48:00:00 script.sh
