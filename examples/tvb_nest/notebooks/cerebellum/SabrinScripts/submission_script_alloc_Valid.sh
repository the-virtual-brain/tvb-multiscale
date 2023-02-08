#!/bin/bash

job_id=$(squeue -n sbi_sample_fit -o %A --noheader)
nodes_info=$(scontrol show -d job $job_id | grep Nodes=n )
nodes_substring=$(echo $nodes_info | cut -d " " -f 1 | cut -c 7-)
node_list=$(scontrol show hostname $nodes_substring | paste -d, -s)
max_jobs_per_node=1

module load sarus
module load daint-mc

export SCRATCH=/scratch/snx3000/souertan
export TVB_ROOT=$SCRATCH/tvb-root
export TVB_MULTISCALE=$SCRATCH/tvb-multiscale
export HOME_DOCKER=/home/docker
export PYTHON=$HOME_DOCKER/env/neurosci/bin/python
export DOCKER_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale
export WORKDIR=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum
export IMAGE=dionperd/tvb-multiscale-dev:parallel_cluster
export SAMPLEFIT=$WORKDIR/cwc_FICfit-validation20jan.py
export SAMPLEFIT_LAUNCHER=$SCRATCH/tvb-multiscale/examples/tvb_nest/notebooks/cerebellum/sbifit_launcher_valid.sh
export LOGS_DIR=$SCRATCH/logs

export LOGS_DIR=$SCRATCH/logs
if [ ! -d $LOGS_DIR ]; then
    mkdir -p $LOGS_DIR
fi

MOUNT_TVB_MULTISCALE="--mount=type=bind,source=${TVB_MULTISCALE},destination=${DOCKER_MULTISCALE}"
MOUNT_TVB_ROOT="--mount=type=bind,source=${TVB_ROOT},destination=${DOCKER_ROOT}"
MOUNT_SAMPLEFIT_LAUNCHER="--mount=type=bind,source=${SAMPLEFIT_LAUNCHER},destination=${SAMPLEFIT_LAUNCHER}"
MOUNT_LOGS_DIR="--mount=type=bind,source=${LOGS_DIR},destination=${LOGS_DIR}"

error=0
node_index=0
num_jobs_on_node=0
Ts=1000
IFS="," read -a node_array <<< $node_list
echo "Number of available nodes: ${#node_array[@]}"
echo "Node list: ${node_array[@]}"
for iG in $(seq 0 3)
do
  

    if (("$num_jobs_on_node" < "$max_jobs_per_node")); then
        num_jobs_on_node=$(expr $num_jobs_on_node + 1)
    else
        node_index=$(expr $node_index + 1)
        num_jobs_on_node=1
    fi

    if (("$node_index" < "${#node_array[@]}")); then 
        echo $node_index
        echo ${node_array[$node_index]} 
        node_id=${node_array[$node_index]}
        echo 'Submitting task for iG='$iG' in allocation '$job_id' on node '$node_id''
        srun -n 1 -N 1 --nodelist=$node_id --jobid=$job_id --exclusive sarus run --entrypoint "" --workdir=$WORKDIR ${MOUNT_TVB_MULTISCALE} ${MOUNT_TVB_ROOT} ${MOUNT_SAMPLEFIT_LAUNCHER} ${MOUNT_LOGS_DIR} ${IMAGE} bash -c "${SAMPLEFIT_LAUNCHER} ${PYTHON} ${SAMPLEFIT} ${LOGS_DIR} ${iG} ${Ts}" &
    else
        echo "Available node number exceeded!"
        error=1
    fi
    if (($error)); then
        break
    fi
    
    if (($error)); then
        break
    fi
done

