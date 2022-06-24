#!/bin/bash

# Have these in your bash_profile and bashrc:
#export SCRATCH=$HOME/scratch  # only for BIH, it is done already
#export TVB=$SCRATCH/Software/TVB
#export TVB_MULTISCALE=$TVB/tvb-multiscale
#export TVB_ROOT=$TVB/tvb-root

module load daint-mc
module load sarus

#pushd ${TVB_MULTISCALE}
#git pull
#popd
#
#pushd ${TVB_ROOT}
#git pull
#popd

export DOCKER_HOME=/home/docker
export DOCKER_TVB_ROOT=${DOCKER_HOME}/packages/tvb-root
export DOCKER_TVB_MULTISCALE=${DOCKER_HOME}/packages/tvb-multiscale
export DOCKER_PYTHON=${DOCKER_HOME}/env/neurosci/bin/python
export DOCKER_WORKDIR=${DOCKER_TVB_MULTISCALE}/examples/tvb_nest/notebooks/cerebellum
export DOCKER_RUN_FILE=${DOCKER_WORKDIR}/scripts.py
export CMD=${DOCKER_PYTHON}' '${DOCKER_RUN_FILE}

export NODES=mc  # or gpu, nothing for BIH cluster
export N_NODES=1
export N_TASKS=1
export JOB=test
export TIME=24:00:00
export OUTPUT_FILE='$SCRATCH/slurm/output.out'
export MAIL=dionperd@gmail.com
export SRUN='srun -A ich012 -C '$NODES' --job-name= '$JOB' --exclusive --nodes '$N_NODES' --ntasks '$N_TASKS' -t '$TIME' --mail-type=ALL --mail-user='$MAIL


# Sarus
## In PizDaint, we assume that these lines have been already ran:
#module load daint-mc # or daint-gpu
#module load sarus
export DOCKER_IMAGE='dionperd/tvb-multiscale-dev:parallel_cluster'

export INTERACTIVE_CMD='sarus --debug run -t --workdir='$DOCKER_WORKDIR'
                              --mount=type=bind,source=${TVB_MULTISCALE},destination=$'DOCKER_TVB_MULTISCALE'
                              --mount=type=bind,source=${TVB_ROOT},destination=$'DOCKER_TVB_ROOT'
                         '$DOCKER_IMAGE' bash'

export SRUN_INTERACTIVE_CMD=$SRUN' --pty -C '$INTERACTIVE_CMD

export COMMAND='sarus --debug run --workdir='$DOCKER_WORKDIR'
                      --mount=type=bind,source=${TVB_MULTISCALE},destination=$'DOCKER_TVB_MULTISCALE'
                      --mount=type=bind,source=${TVB_ROOT},destination=$'DOCKER_TVB_ROOT'
                '$DOCKER_IMAGE' '$CMD

export SRUN_COMMAND=$SRUN' '$COMMAND

$SRUN_COMMAND

## Singularity
### In PizDaint, we assume that these lines have been already ran:
##module load daint-mc # or daint-gpu
##module load singularity/3.5.3-daint
## singularity pull docker://${DOCKER_IMAGE}
#export DOCKER_IMAGE_RUN='tvb-nest_dp-ongoing-work.sif'
#
## Run from within the path where the .sif file is:
#
#export INTERACTIVE_CMD='singularity shell
#                        --bind $TVB_MULTISCALE':$DOCKER_TVB_MULTISCALE'
#                        --bind $TVB_ROOT:'$DOCKER_TVB_ROOT'
#                        '$DOCKER_IMAGE_RUN
#
#export SRUN_INTERACTIVE_CMD=$SRUN' -pty -C '$INTERACTIVE_CMD
#
#export COMMAND='singularity exec
#                --bind $TVB_MULTISCALE:'$DOCKER_TVB_MULTISCALE'
#                --bind $TVB_ROOT:'$DOCKER_TVB_ROOT'
#                '$DOCKER_IMAGE' '$CMD
#
#export SRUN_COMMAND=$SRUN' '$COMMAND



