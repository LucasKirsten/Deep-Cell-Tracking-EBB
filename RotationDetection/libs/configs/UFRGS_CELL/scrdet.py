# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.faster_rcnn_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 0.001 * BATCH_SIZE * NUM_GPU
SAVE_WEIGHTS_INTE = 10000
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'UFRGS_CELL'
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 2

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# neck
FPN_MODE = 'scrdet'

# rpn head
USE_CENTER_OFFSET = False
BASE_ANCHOR_SIZE_LIST = 256
ANCHOR_STRIDE = 8
ANCHOR_SCALES = [0.0625, 0.125, 0.25, 0.5, 1., 2.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 10.0]
ANCHOR_SCALE_FACTORS = None
ANCHOR_MODE = 'H'
ANGLE_RANGE = 90

# loss
USE_IOU_FACTOR = False

# post-processing
FAST_RCNN_H_NMS_IOU_THRESHOLD = 0.3
FAST_RCNN_R_NMS_IOU_THRESHOLD = 0.2

VERSION = 'SCRDet_UFRGS_CELL_smooth_l1_loss'