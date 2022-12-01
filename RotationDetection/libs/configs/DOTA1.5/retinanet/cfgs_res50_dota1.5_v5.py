# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SAVE_WEIGHTS_INTE = 32000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA1.5'
CLASS_NUM = 16

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = 1  # IoU-Smooth L1

VERSION = 'RetinaNet_DOTA1.5_2x_20210418'

"""
RetinaNet-H + IoU-Smooth L1
FLOPs: 862193566;    Trainable params: 33051321

This is your evaluation result for task 1:

    mAP: 0.591721330560492
    ap of each class:
    plane:0.7916196205179232,
    baseball-diamond:0.7247341048049472,
    bridge:0.39807651015690093,
    ground-track-field:0.6420371678606354,
    small-vehicle:0.4694358822053647,
    large-vehicle:0.5301717570501657,
    ship:0.7362725830519496,
    tennis-court:0.8960018234487599,
    basketball-court:0.7216228096733761,
    storage-tank:0.5939637375904644,
    soccer-ball-field:0.5020787833793735,
    roundabout:0.6553347144611664,
    harbor:0.5372875179382095,
    swimming-pool:0.6467734934896577,
    helicopter:0.5303522112203418,
    container-crane:0.09177857211863588

The submitted information is :

Description: RetinaNet_DOTA1.5_2x_20210418_108.8w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


