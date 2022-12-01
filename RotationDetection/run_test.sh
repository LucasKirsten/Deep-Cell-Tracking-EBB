#!/bin/bash

DATASET=$1
LINEAGE=$2
EXT=$3

cd /workdir/SW/RotationDetection/tools/r2cnn
python test_ufrgscell.py --img_dir=/workdir/$DATASET/$LINEAGE  \
                          --gpu=0 \
                          --image_ext=$EXT \
                          --test_annotation_path=/workdir/$DATASET/$LINEAGE/xml_rotdet \
                          none
