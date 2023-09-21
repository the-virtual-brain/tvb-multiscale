#!/bin/bash

# Have these in your bash_profile and bashrc:
#export SCRATCH=$HOME/scratch  # only for BIH, it is done already
#export TVB=$SCRATCH/Software/TVB
#export TVB_MULTISCALE=$TVB/tvb-multiscale
#export TVB_ROOT=$TVB/tvb-root

export DOCKER_HOME=/home/docker
export DOCKER_TVB_ROOT=$DOCKER_HOME/packages/tvb-root
export DOCKER_TVB_MULTISCALE=$DOCKER_HOME/packages/tvb-multiscale
export DOCKER_WORKDIR=$DOCKER_TVB_MULTISCALE/tvb_multiscale/examples/tvb_nest
export DOCKER_PYTHON=/usr/bin/python
export DOCKER_RUN_FILE=/home/docker/packages/tvb-multiscale/tvb_multiscale/examples/tvb_nest/example.py
export CMD=$DOCKER_PYTHON' '$DOCKER_RUN_FILE

export NODES=mc  # or gpu, nothing for BIH cluster
export TIME=24:00:00
export OUTPUT_FILE='$SCRATCH/slurm/output.out'
export MAIL=...@gmail.con
export SRUN='srun -C '$NODES' -t '$TIME' -o '$OUTPUT_FILE' --mail-type=ALL --mail-user='$MAIL


# Sarus
## In PizDaint, we assume that these lines have been already ran:
#module load daint-mc # or daint-gpu
#module load sarus
export DOCKER_IMAGE='thevirtualbrain/tvb-nest:dev'

export INTERACTIVE_CMD='sarus --debug run -t --mpi --workdir='$DOCKER_WORKDIR'
                              --mount=type=bind,source=${TVB_MULTISCALE},destination=$'DOCKER_TVB_MULTISCALE'
                              --mount=type=bind,source=${TVB_ROOT},destination=$'DOCKER_TVB_ROOT'
                         '$DOCKER_IMAGE' bash'

export SRUN_INTERACTIVE_CMD=$SRUN' --pty -C '$INTERACTIVE_CMD

export COMMAND='sarus --debug run --mpi --workdir='$DOCKER_WORKDIR'
                      --mount=type=bind,source=${TVB_MULTISCALE},destination=$'DOCKER_TVB_MULTISCALE'
                      --mount=type=bind,source=${TVB_ROOT},destination=$'DOCKER_TVB_ROOT'
                '$DOCKER_IMAGE' '$CMD

export SRUN_COMMAND=$SRUN' '$COMMAND

## Singularity
### In PizDaint, we assume that these lines have been already ran:
##module load daint-mc # or daint-gpu
##module load singularity/3.5.3-daint
## singularity pull docker://${DOCKER_IMAGE}
#export DOCKER_IMAGE_RUN='tvb-nest_dev.sif'
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



