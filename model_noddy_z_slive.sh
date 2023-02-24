#!/bin/bash

#SBATCH --job-name=remodel_noddy
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --partition=short
module list

### Luke Smith 2023
# I downloaded a set of published Noddy .his files
# We will use an independent Slurm job for each .his to be processed.
# For each job/file;
# Increment the geophysics altitude by 20 m, then
# forward model TMI, then
# move resulting grid to long term storage, until
# 4000 m altitude.

# 5 runs / minute 16C8GB

HISGZ_FILE=$1
SCRATCH="$MYSCRATCH/$SLURM_JOBID/"
RESULTS="${MYGROUP}/results/${HISGZ_FILE%%.*}/"
OVERALL_LOG="${RESULTS}"/overall_log.txt
HIS_NAME="${HISGZ_FILE%%.*}"
OUTPUT="${HIS_NAME}"
TEMP_HIS="temp.his"

echo "Processing ${HISGZ_FILE}"
mkdir -p "${SCRATCH}"
mkdir -p "${RESULTS}"
cp "${HISGZ_FILE}" "${SCRATCH}"
cd "${SCRATCH}"
pigz -dk "${HISGZ_FILE}"
echo "SCRATCH is ${SCRATCH}, RESULTS is ${RESULTS}"

for ALT in {0100..4000..20}; do
  sed /Altitude/c\ "\        Altitude\        = ${ALT}.00" "${HIS_NAME}.his" \
    > "${TEMP_HIS}"

  /group/cet001/shared/noddy "${TEMP_HIS}" "${OUTPUT}_${ALT}m.his" GEOPHYSICS
  mv *.mag "${RESULTS}"
  mv *.grv "${RESULTS}"

  echo "Processed ${HISGZ_FILE} at altitude ${ALT} and moved to ${RESULTS}"

done
