#!/bin/bash

DATASET="PhC-C2DH-U373"
LINEAGE="02"
AUGMENT=1
FROM_CROPS=0

bash ISBI_inferece.sh ${DATASET} ${LINEAGE} ${AUGMENT} ${FROM_CROPS}
