#!/bin/bash -l
#SBATCH --job-name="sbi_fit_test"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dionperd@gmail.com
#SBATCH --time=00:60:00
#SBATCH --nodes=10
#SBATCH --ntasks=1
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=1G


export WORK=/data/gpfs-1/users/perdikid_c/work
#export TVB_ROOT=$WORK/tvb-root
# export TVB_MULTISCALE=$$WORK/tvb-multiscale
export HOME_DOCKER=/home/docker
export PYTHON=$HOME_DOCKER/env/neurosci/bin/python
export DOCKER_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale
export WORKDIR=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum
export IMAGE=$WORK/tvb_multi_dev_parallel_cluster.sif
export SBIFIT=$WORKDIR/scripts.py
export G=1

singularity exec --pwd $WORKDIR --bind $TVB_ROOT/:$DOCKER_ROOT,$TVB_MULTISCALE/:$DOCKER_MULTISCALE $IMAGE $PYTHON $SBIFIT $G

# run it with
# sbatch -e errors.txt -o outputs.txt run_script_bih_singularity.sh

# instead, for interactive run:
# srun --time 1-00 --mem=16G --ntasks=16 --pty bash -i
# singularity run --pwd $WORKDIR --bind $TVB_ROOT/:$DOCKER_ROOT,$TVB_MULTISCALE/:$DOCKER_MULTISCALE $IMAGE