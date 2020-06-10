#!/bin/bash

# Have these in your bash_profile and bashrc:
#export SCRATCH=$HOME/scratch  # only for BIH, it is done already
#export TVB=$SCRATCH/Software/TVB
#export TVB_MULTISCALE=$TVB/tvb-multiscale
#export TVB_ROOT=$TVB/tvb-root

export DOCKER_HOME=/home/docker
export DOCKER_TVB_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_TVB_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale

export DOCKER_IMAGE='thevirtualbrain/tvb-nest:dp-ongoing-work'
export CMD="/home/docker/env/neurosci/bin/python /home/docker/packages/tvb-multiscale/tvb_nest/examples/example.py"

export NODES=mc  # or gpu, nothing for BIH cluster
export SRUN='srun -C $NODES'

# Sarus
## In PizDaint, we assume that these lines have been already ran:
#module load daint-mc # or daint-gpu
#module load sarus

export INTERACTIVE_CMD='sarus --debug run -t
                        --mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_TVB_MULTISCALE}
                        --mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_TVB_ROOT}
                         $DOCKER_IMAGE bash'

export SRUN_INTERACTIVE_CMD='$SRUN --pty -C ${INTERACTIVE_CMD}'

export COMMAND='sarus --debug run
                --mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_TVB_MULTISCALE}
                --mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_TVB_ROOT}
                $DOCKER_IMAGE $CMD'

export SRUN_COMMAND='$SRUN $COMMAND'

# Singularity
## In PizDaint, we assume that these lines have been already ran:
#module load daint-mc # or daint-gpu
#module load singularity/3.5.3-daint

export INTERACTIVE_CMD='singularity shell
                        --bind ${TVB_MULTISCALE}:${DOCKER_TVB_MULTISCALE}
                        --bind ${TVB_ROOT}:${DOCKER_TVB_ROOT}
                        docker://${DOCKER_IMAGE}'

export SRUN_INTERACTIVE_CMD='$SRUN -pty -C ${INTERACTIVE_CMD}'

export COMMAND='singularity exec
                --bind ${TVB_MULTISCALE}:${DOCKER_TVB_MULTISCALE}
                --bind ${TVB_ROOT}:${DOCKER_TVB_ROOT}
                docker://${DOCKER_IMAGE} $CMD'

export SRUN_COMMAND='$SRUN $COMMAND'



