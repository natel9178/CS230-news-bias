#!/bin/sh
module load cudnn
sbatch --partition=gpu --gres=gpu:1 --qos=gpu --time=48:00:00 script.sh
