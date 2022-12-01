# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from libs.configs._base_.models.retinanet_r50_fpn import *
from libs.configs._base_.datasets.dota_detection import *
from libs.configs._base_.schedules.schedule_1x import *
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

# schedule
BATCH_SIZE = 1
GPU_GROUP = "0,1"
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

# bbox head
ANGLE_RANGE = 180

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA1.5_2x_20210424'

"""
RetinaNet-H + theta=atan(sin(theta)/cos(theta))
This is your evaluation result for task 1:

    mAP: 0.5824908699297879
    ap of each class:
    plane:0.7895020428660102,
    baseball-diamond:0.7426283801990676,
    bridge:0.4042942801329716,
    ground-track-field:0.638442832172445,
    small-vehicle:0.40612339237497663,
    large-vehicle:0.4665677105035402,
    ship:0.66779870860629,
    tennis-court:0.8934470198741159,
    basketball-court:0.7069387566887745,
    storage-tank:0.5870570591107861,
    soccer-ball-field:0.502685240006103,
    roundabout:0.6758452798646649,
    harbor:0.5356213731476638,
    swimming-pool:0.645343191971053,
    helicopter:0.5461950149945082,
    container-crane:0.11136363636363636

The submitted information is :

Description: RetinaNet_DOTA1.5_2x_20210424_83.2w
Username: AICyber
Institute: IECAS
Emailadress: yangxue16@mails.ucas.ac.cn
TeamMembers: Yang Xue; Yang Jirui
"""



