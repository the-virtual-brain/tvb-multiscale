#!/bin/bash -l
#SBATCH --job-name="sbi_fit"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dionperd@gmail.com
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=70
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=1G

module load sarus
module load daint-mc

# export SCRATCH=/scratch/snx3000/bp000229
#export TVB_ROOT=$SCRATCH/Software/TVB/tvb-root
#export TVB_MULTISCALE=$SCRATCH/Software/TVB/tvb-multiscale
export HOME_DOCKER=/home/docker
export PYTHON=$HOME_DOCKER/env/neurosci/bin/python
export DOCKER_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale
export WORKDIR=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum
export IMAGE=dionperd/tvb-multiscale-dev:parallel_cluster
export SBIFIT=$WORKDIR/scripts/sbi_script.py

for iG in $(seq 0 6)
do
  for iB in $(seq 0 9)
  do
    echo 'Submitting task for iG='$iG', and iB='$iB'...'
    sarus run --entrypoint "" --workdir=$WORKDIR --mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_MULTISCALE} --mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_ROOT} $IMAGE $PYTHON ${SBIFIT} -scr 0 -ig $iG -ib $iB &
  done
done

wait

echo "Job done..."
# run it with
# sbatch -A ich012 -e errors_iG0-2.txt -o outputs_iG0-2.txt run_batch_cscs_sarus_iG0-2.sh
