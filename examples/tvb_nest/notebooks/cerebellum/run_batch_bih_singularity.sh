#!/bin/bash -l
#SBATCH --job-name="sbi_fit_normal"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dionperd@gmail.com
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=3G

export WORK=/data/gpfs-1/users/perdikid_c/work
#export TVB_ROOT=$WORK/tvb-root
# export TVB_MULTISCALE=$$WORK/tvb-multiscale
export HOME_DOCKER=/home/docker
export PYTHON=$HOME_DOCKER/env/neurosci/bin/python
export DOCKER_ROOT=$HOME_DOCKER/packages/tvb-root
export DOCKER_MULTISCALE=$HOME_DOCKER/packages/tvb-multiscale
export WORKDIR=$DOCKER_MULTISCALE/examples/tvb_nest/notebooks/cerebellum
export IMAGE=$WORK/tvb_multi_dev_parallel_cluster.sif
export SBIFIT=$WORKDIR/scripts/sbi_script.py

for iG in $(seq 0 2)
do
  for iB in $(seq 0 9)
  do
    echo 'Submitting task for iG='$iG', and iB='$iB'...'
    singularity exec --pwd $WORKDIR --bind $TVB_ROOT/:$DOCKER_ROOT,$TVB_MULTISCALE/:$DOCKER_MULTISCALE $IMAGE $PYTHON $SBIFIT -scr 0 -ig $iG -ib $iB &
  done
done

wait

echo "Job done..."

# run it with
# sbatch -e errors.txt -o outputs.txt run_batch_bih_singularity.sh
