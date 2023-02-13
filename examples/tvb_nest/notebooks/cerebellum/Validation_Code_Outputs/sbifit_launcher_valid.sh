#!/bin/bash

PYTHON=$1
SAMPLEFIT=$2
LOGS_DIR=$3
iG=$4
Ts=$5

${PYTHON} ${SAMPLEFIT} -scr 0 -ig ${iG} -ts ${Ts} 1> ${LOGS_DIR}/iG${iG}Ts${Ts}.out 2> ${LOGS_DIR}/iG${iG}Ts${Ts}.err

