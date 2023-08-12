#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c4
#SBATCH --gres=gpu:1
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job%j.out

CONFIG_FILE=$1

echo $CONFIG_FILE

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

python ME/train/train_sigmask.py --valid_iter 5000 --print_iter 500 --plot_iter 2000 $CONFIG_FILE

