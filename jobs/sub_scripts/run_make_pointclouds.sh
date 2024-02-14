#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -c4
#SBATCH -J make_larnd_voxelmaps
#SBATCH --array 1-10
#SBATCH --mem=10000
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job%j.out

INPUT_DIR=$1
OUTPUT_DIR=$2
VMAP_PATH=$3

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "input dir is ${INPUT_DIR}"
echo "output dir is ${OUTPUT_DIR}"
echo "voxel map is ${VMAP_PATH}"

input_name=$(ls $INPUT_DIR | head -n $SLURM_ARRAY_TASK_ID | tail -n -1)
input_file=${INPUT_DIR}/${input_name}

echo "input file is ${input_file}"

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

python data_scripts/make_larnd_pointclouds.py --batch_mode $input_file $OUTPUT_DIR $VMAP_PATH

