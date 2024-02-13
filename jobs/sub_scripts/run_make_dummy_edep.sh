#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -c2
#SBATCH --mem=10000
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job%j.out

INPUT_PATH=$1
OUTPUT_PATH=$2
MODE=$3

echo "Job id ${SLURM_JOB_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "input path is ${INPUT_PATH}"
echo "output path is ${OUTPUT_PATH}"
echo "mode is ${MODE} (1: fix_x and fix_z)"

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

if [[ "$MODE" == 1 ]]; then
  python data_scripts/make_dummy_edep.py --batch_mode \
                                         --fix_x \
                                         --fix_y \
                                         $INPUT_PATH \
                                         $OUTPUT_PATH
else
  echo "invalid mode (${MODE})"
  exit 1
fi

