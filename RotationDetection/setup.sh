#!/bin/bash

cd /workdir/msc/RotationDetection/libs/utils/cython_utils
python setup.py build_ext --inplace

cd /workdir/msc/RotationDetection/libs/utils/
python setup.py build_ext --inplace