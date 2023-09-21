#!/bin/sh

echo "Pulling TVB-Multiscale docker image"
/apps/daint/system/opt/sarus/1.1.0/bin/sarus pull thevirtualbrain/tvb-multiscale:2.1.0

start=$SECONDS

echo "Start docker container"
srun -C mc /apps/daint/system/opt/sarus/1.1.0/bin/sarus --debug run --mpi thevirtualbrain/tvb-multiscale:2.1.0 /home/docker/.local/bin/python /home/docker/packages/tvb-multiscale/docker/installation_test.py

duration=$(( SECONDS - start ))

echo "TVB-NEST test completed in $duration seconds"