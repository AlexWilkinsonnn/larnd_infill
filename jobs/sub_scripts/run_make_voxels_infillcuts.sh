#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -c4
#SBATCH -J make_larnd_voxelmaps
#SBATCH --array 1-10
#SBATCH --mem=10000
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job.%x.%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job.%x.%j.out

INPUT_DIR=$1
OUTPUT_DIR=$2
VMAP_PATH=$3
MODE=$4

echo "Job id ${SLURM_JOB_ID}"
echo "Job array task id ${SLURM_ARRAY_TASK_ID}"
echo "Running on ${SLURM_JOB_NODELIST}"
echo "input dir is ${INPUT_DIR}"
echo "output dir is ${OUTPUT_DIR}"
echo "voxel map is ${VMAP_PATH}"
echo "mode is ${MODE} (1: --forward_facing_anode_zshift 0.38 --backward_facing_anode_zshift -0.38 --infillinfo_cuts)"

input_name=$(ls $INPUT_DIR | head -n $SLURM_ARRAY_TASK_ID | tail -n -1)
input_file=${INPUT_DIR}/${input_name}

echo "input file is ${input_file}"

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

if [[ "$MODE" == 1 ]]; then
  python data_scripts/make_larnd_voxels_fromp3d.py --batch_mode \
                                                   --forward_facing_anode_zshift 0.38 \
                                                   --backward_facing_anode_zshift -0.38 \
                                                   $input_file \
                                                   $OUTPUT_DIR \
                                                   $VMAP_PATH
else
  echo "invalid mode (${MODE})"
  exit 1
fi

