#!/bin/bash

DATASET=$1
LINEAGE=$2
AUGMENT=$3
FROM_CROPS=$4

# activate conda env
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate lkirstenisbi

# convert tif images to jpg
cd /workdir/SW/RotationDetection/dataloader/dataset/UFRGS_CELL
python convert2jpg.py /workdir/${DATASET}/${LINEAGE} .tif ${AUGMENT}

# handle cases with image augmentation
if [ "$FROM_CROPS" -eq 1 ]; then
    # make crops of the images
    cd /workdir/SW/RotationDetection/dataloader/dataset/UFRGS_CELL
    python image_crop.py /workdir/${DATASET}/${LINEAGE}

    # run model inference
    cd /workdir/SW/RotationDetection
    python ISBI_run_inference.py --img_dir=/workdir/${DATASET}/${LINEAGE}/crop --image_ext='.png' ${DATASET}
else
    # run model inference
    cd /workdir/SW/RotationDetection
    python ISBI_run_inference.py --img_dir=/workdir/${DATASET}/${LINEAGE} --image_ext='.jpg' ${DATASET}
fi

# rename results
mv /workdir/SW/RotationDetection/${DATASET}_results/det_normal_cell.txt \
   /workdir/SW/RotationDetection/${DATASET}_results/det_mitoses.txt
echo "" > /workdir/SW/RotationDetection/${DATASET}_results/det_normal_cell.txt

# run tracker
cd /workdir/SW/track_by_detection
cp ./configs/${DATASET}_${LINEAGE}.py ./tracker/configs.py
python ISBI_run.py True
