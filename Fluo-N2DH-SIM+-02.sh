#!/bin/bash

DATASET="Fluo-N2DH-SIM+"
LINEAGE="02"
AUGMENT=1
FROM_CROPS=1

bash ISBI_inferece.sh ${DATASET} ${LINEAGE} ${AUGMENT} ${FROM_CROPS}
