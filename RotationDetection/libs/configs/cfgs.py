# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 8
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 2985//BATCH_SIZE
MAX_EPOCH = 100
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'UFRGS_CELL_1class'
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 1
MEAN = (17.05+23.09)/2.
STD  = (11.81+7.32)/2.
PIXEL_MEAN = [MEAN]*3
PIXEL_MEAN_ = [255./MEAN]*3
PIXEL_STD = [255./STD]*3

# data augmentation
IMG_ROTATE = True
RGB2GRAY = True
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# backbone
pretrain_zoo = PretrainModelZoo()
DATASET_PATH = '/workdir'
ROOT_PATH = '/workdir/SW/RotationDetection'
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
SUMMARY_PATH = os.path.join(ROOT_PATH, 'output/summary')

VERSION = 'R2CNN_Fluo-N2DH-SIM+-all_smooth_l1_loss'