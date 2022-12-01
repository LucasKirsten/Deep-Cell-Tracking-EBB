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
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0
REG_LOSS_MODE = None

VERSION = 'RetinaNet_DOTA_2x_20201107'

"""
RSDet-5p
FLOPs: 860451115;    Trainable params: 33002916
This is your result for task 1:

mAP: 0.6605200411622921
ap of each class: plane:0.8870654106758245,
baseball-diamond:0.7268503783923166,
bridge:0.4291863271038613,
ground-track-field:0.6491441522226533,
small-vehicle:0.6791811400732405,
large-vehicle:0.5291229360078313,
ship:0.7267297470095706,
tennis-court:0.9025198415246758,
basketball-court:0.7698198708926544,
storage-tank:0.8022000971627501,
soccer-ball-field:0.5130537969212124,
roundabout:0.5820649854991844,
harbor:0.5364528563483577,
swimming-pool:0.672263075454247,
helicopter:0.5021460021460021

The submitted information is :

Description: RetinaNet_DOTA_2x_20201107_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue
"""


