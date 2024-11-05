#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 60
#SBATCH -J infill
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s"
#SBATCH --array=1-181
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/%x.%A_%a.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/%x.%A_%a.out

################################################################################
# Options

SCRATCH_DIR="/state/partition1/awilkins/scratch/${SLURM_JOB_ID}"

CONFIG_FILE=$1
INPUT_DIR=$2
OUTPUT_DIR=$3

################################################################################

mkdir -p ${SCRATCH_DIR}

input_name=$(ls $INPUT_DIR | head -n $SLURM_ARRAY_TASK_ID | tail -n -1)
input_file=${INPUT_DIR}/${input_name}
output_name=${input_name%.*}_infilled.h5
output_file=${SCRATCH_DIR}/${output_name}

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "With cuda device ${CUDA_VISIBLE_DEVICES}"
echo "Input file is ${input_file}"
echo "Output file will be ${output_file_final}"

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

python run_infill.py --input_file $input_file \
                     --output_file $output_file \
                     --cache_dir $SCRATCH_DIR \
                     $CONFIG_FILE

if [[ $? == 0 ]]
then
  cp ${SCRATCH_DIR}/${output_name} ${OUTPUT_DIR}/${output_name}
else
  echo "Python script exited badly! not copying $output_name to $OUTPUT_DIR"
fi

rm -r ${SCRATCH_DIR}
