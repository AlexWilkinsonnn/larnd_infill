#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -c4
#SBATCH --mem=10000
#SBATCH --nodelist=compute-0-18
#SBATCH --error=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/err/job%j.err
#SBATCH --output=/home/awilkins/larnd_infill/larnd_infill/jobs/logs/out/job%j.out

INPUT_DIR=$1
OUTPUT_DIR=$2
VMAP_PATH=$3
SMEARZ=$4

echo $INPUT_DIR
echo $OUTPUT_DIR
echo $VMAP_PATH
echo $SMEARZ

# Make sure they have trailing forward slashes
# INPUT_DIR=$(echo $INPUT_DIR | sed 's#[^/]$#&/#')
# OUTPUT_DIR=$(echo $OUTPUT_DIR | sed 's#[^/]$#&/#')

cd /home/awilkins/larnd_infill/larnd_infill
source setups/setup.sh

for file in ${INPUT_DIR%/}/*; do
  echo $file
  python data_scripts/make_larnd_pointclouds.py --batch_mode \
                                                --smear_z $SMEARZ \
                                                $file \
                                                $OUTPUT_DIR \
                                                $VMAP_PATH
done

