#!/bin/bash

# verify if git repository is on system
if [[ ! -d ./RotationDetection/ ]]
then
    echo "Downloading github repository..."
    git clone https://github.com/LucasKirsten/RotationDetection.git
fi

# checkout to correct branch and update repository
cd RotationDetection
git checkout cell_detector
git pull --force

# ask for images path
echo "Input the path for the images"
read IMG_PATH

# ask for model to make predictions
echo "Input the model to make predictions (type the number):
[1] CSL
[2] DCL
[3] R3Det
[4] RetinaNet"
read MODEL

# function to download weights
dowload_weights() {
    fileid=$1
    filename=$2
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
    unzip $filename
    rm $filename
}

# verify model weights
if [[ $MODEL -eq 1 ]]
then
    MODEL_NAME='csl'
    if [[ ! -d output/trained_weights/CSLv1_UFRGS_CELL_smooth_l1_loss ]]
    then
        echo "Downloading weights for CLS model..."
        cd output/trained_weights
        dowload_weights "1s4AVYlGh3zd6EOg2BzDtsmfZYC5dAcf9" "CSLv1_UFRGS_CELL_smooth_l1_loss.zip"
        cd ../..
    fi
elif [[ $MODEL -eq 2 ]]
then
    MODEL_NAME='dcl'
    if [[ ! -d output/trained_weights/DLCv3_UFRGS_CELL_smooth_l1_loss ]]
    then
        echo "Downloading weights for DCL model..."
        cd output/trained_weights
        dowload_weights "1KVM2vCB73HPxrhZq52lC6YF4a3HIymG9" "DLCv3_UFRGS_CELL_smooth_l1_loss.zip"
        cd ../..
    fi
elif [[ $MODEL -eq 3 ]]
then
    MODEL_NAME='r3det'
    if [[ ! -d output/trained_weights/R3Det_UFRGS_CELL_smooth_l1_loss ]]
    then
        echo "Downloading weights for R3Det model..."
        cd output/trained_weights
        dowload_weights "1Jdsa93O-dx4QPt0qc7XyKB2YvJUB51kg" "R3Det_UFRGS_CELL_smooth_l1_loss.zip"
        cd ../..
    fi
elif [[ $MODEL -eq 4 ]]
then
    MODEL_NAME='retinanet'
    if [[ ! -d output/trained_weights/RetinaNet_UFRGS_CELL_smooth_l1_loss ]]
    then
        echo "Downloading weights for RetinaNet model..."
        cd output/trained_weights
        dowload_weights "1IZrUGrtpEJM9FGh0vY8LfrQGR6aa0_dd" "RetinaNet_UFRGS_CELL_smooth_l1_loss.zip"
        cd ../..
    fi
else
    echo "Invalid model!"
    exit 0
fi

# run docker
docker run -it --name uranusdet -v $(pwd):/workdir/msc/RotationDetection -v $IMG_PATH:/workdir/msc/dataset -w /workdir/msc/RotationDetection --gpus all -d lucasnkirsten/uranusdet:latest
# run inference inside docker image
docker exec uranusdet bash setup.sh
docker exec uranusdet python run_inference.py --img_dir='/workdir/msc/dataset' --gpu=0 --image_ext='.jpg' $MODEL_NAME

# remove container (if still running)
docker stop uranusdet && docker rm uranusdet