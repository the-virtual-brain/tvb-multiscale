#!/bin/bash -l
#SBATCH --job-name="sbi_fit_test"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dionperd@gmail.com
#SBATCH --time=00:60:00
#SBATCH --nodes=10
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
export HOME_DOCKER=/home/docker
export PYTHON=$HOME_DOCKER/env/neurosci/bin/python
export DOCKER_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale
export WORKDIR=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum
export IMAGE=dionperd/tvb-multiscale-dev:parallel_cluster
export SBIFIT=$WORKDIR/scripts.py
export G=1

sarus run --workdir=$WORKDIR --mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_MULTISCALE} --mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_ROOT} $IMAGE $PYTHON ${SBIFIT} $G

# run it with
# sbatch -A ich012 -e errors.txt -o outputs.txt run_script_cscs_sarus.sh

# instead, for interactive run:
# srun -C mc -A ich012 --time 1-00 --mem=16G --ntasks=16 --pty bash -i
# sarus run -t --workdir=$WORKDIR --mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_MULTISCALE} --mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_ROOT} $IMAGE bash
