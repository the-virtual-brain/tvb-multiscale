#!/bin/bash

PYTHON=$1
SBIFIT=$2
LOGS_DIR=$3
iG=$4
iB=$5

${PYTHON} ${SBIFIT} -scr 0 -ig ${iG} -ib ${iB} 1> ${LOGS_DIR}/iG${iG}iB${iB}.out 2> ${LOGS_DIR}/iG${iG}iB${iB}.err

