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
REG_WEIGHT = 1.0
ANGLE_WEIGHT = 0.5
REG_LOSS_MODE = None

# CSL
LABEL_TYPE = 0  # {0: gaussian_label, 1: rectangular_label, 2: pulse_label, 3: triangle_label}
RADUIUS = 6
OMEGA = 1

VERSION = 'RetinaNet_DOTA1.5_CSL_2x_20210430'

"""
gaussian label, omega=1, r=6
FLOPs: 909213886;    Trainable params: 41764221
This is your evaluation result for task 1:

mAP: 0.5775673470648883
ap of each class: plane:0.7811738810693274, baseball-diamond:0.7340045973013912, bridge:0.39600999422871364, ground-track-field:0.6200485132756077, small-vehicle:0.4640935125124646, large-vehicle:0.4856814068413247, ship:0.7232416775725545, tennis-court:0.8972758770380962, basketball-court:0.7164166681737884, storage-tank:0.5785528515150301, soccer-ball-field:0.46354461303899874, roundabout:0.6506594270519598, harbor:0.5344306300908814, swimming-pool:0.632588906847701, helicopter:0.45091480509281395, container-crane:0.11244019138755981
The submitted information is :

Description: RetinaNet_DOTA1.5_CSL_2x_20210430_83.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""


