#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -c2
#SBATCH --mem=10000
#SBATCH --array=0-22
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job%j.out

OUTPUT_DIR=$1
VMAP_PATH=$2
N=$3

START_IDX=$((${N}*${SLURM_ARRAY_TASK_ID}))

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "output dir is ${OUTPUT_DIR}"
echo "voxel map is ${VMAP_PATH}"
echo "number to process is ${N}"
echo "starting index is ${START_IDX}"

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

python data_scripts/make_dummy_larnd_pointclouds.py --batch_mode \
                                                    --start_index $START_IDX \
                                                    $OUTPUT_DIR \
                                                    $VMAP_PATH \
                                                    $N

