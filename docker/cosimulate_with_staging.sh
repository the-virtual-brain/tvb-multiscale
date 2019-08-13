#!/bin/sh
. /opt/cray/pe/modules/3.2.10.6/init/bash
. /etc/profile.d/cscs.sh
module load /apps/daint/UES/easybuild/modulefiles/daint-mc
module load /apps/daint/system/modulefiles/shifter-ng/18.06.0
start=$SECONDS

shifter run --mpi --mount=type=bind,source=$HOME,destination=$HOME thevirtualbrain/tvb-nest /home/docker/env/neurosci/bin/python $1 $2

duration=$(( SECONDS - start ))
echo "TVB-NEST test completed in $duration seconds"