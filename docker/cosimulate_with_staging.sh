#!/bin/sh
. /opt/cray/pe/modules/3.2.11.3/init/bash
. /etc/profile.d/cscs.sh
module load sarus
start=$SECONDS

sarus run --mpi --mount=type=bind,source=$HOME,destination=$HOME thevirtualbrain/tvb-nest:0.3 /home/docker/env/neurosci/bin/python $1 $2

duration=$(( SECONDS - start ))
echo "TVB-NEST test completed in $duration seconds"