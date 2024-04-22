#!/bin/bash
#SBATCH -p GPU
#SBATCH -N1
#SBATCH -c4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s"
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job.%x.%j.out

CONFIG_FILE=$1

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
echo $CONFIG_FILE

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

python ME/train/train_sigmask_adversarial.py --print_iter 100 --plot_iter -1 $CONFIG_FILE

