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


# export SCRATCH=/scratch/snx3000/bp000229
#export TVB_ROOT=$SCRATCH/Software/TVB/tvb-root
#export TVB_MULTISCALE=$SCRATCH/Software/TVB/tvb-multiscale
export HOME_DOCKER=/home/docker
export PYTHON=$HOME_DOCKER/env/neurosci/bin/python
export DOCKER_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale
export WORKDIR=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum
export IMAGE=tvb_multiscale_dev_parallel_cluster
export SBIFIT=$WORKDIR/scripts.py
export G=1

singularity exec --pwd $WORKDIR --bind $TVB_ROOT/:$DOCKER_ROOT,$TVB_MULTISCALE/:$DOCKER_MULTISCALE $IMAGE $PYTHON $SBIFIT %G