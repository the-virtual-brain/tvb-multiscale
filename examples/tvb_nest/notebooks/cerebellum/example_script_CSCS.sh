#!/bin/bash -l
#SBATCH --job-name="sbi_fit_test"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dionperd@gmail.com
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=1G

module load sarus
module load daint-mc

# export SCRATCH=/scratch/snx3000/bp000229
#export TVB_ROOT=$SCRATCH/Software/TVB/tvb-root
#export TVB_MULTISCALE=$SCRATCH/Software/TVB/tvb-multiscale
export DOCKER_ROOT=/home/docker/packages/tvb-root
export DOCKER_MULTISCALE=/home/docker/packages/tvb-multiscale
export SBIFIT=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum/scripts.py
export G=1
export PYTHON=/home/docker/env/neurosci/bin/python
export IMAGE=dionperd/tvb-multiscale-dev:parallel_cluster

sarus run --mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_MULTISCALE} --mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_ROOT} $IMAGE $PYTHON ${SBIFIT} $G
