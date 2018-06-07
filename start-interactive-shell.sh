#!/bin/bash
module load cudnn
srun --pty --partition=gpu --gres=gpu:1 --qos=interactive $SHELL -l