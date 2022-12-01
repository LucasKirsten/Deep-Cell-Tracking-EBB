# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
RSDet-8p  20epoch
FLOPs: 865678510;    Trainable params: 33148131
{'0.8': {'roundabout': 0.295204242581896, 'baseball-diamond': 0.22544820697152898, 'swimming-pool': 0.10269934843313683, 'helicopter': 0.1270053475935829, 'ground-track-field': 0.30959819095145613, 'basketball-court': 0.44296886891823595, 'storage-tank': 0.4444430393972967, 'large-vehicle': 0.14058332918456087, 'small-vehicle': 0.17658342417200618, 'mAP': 0.30596702056936526, 'soccer-ball-field': 0.4373140805862623, 'harbor': 0.11112165496299109, 'plane': 0.6416279801778162, 'bridge': 0.04178018078433016, 'tennis-court': 0.7888738557403647, 'ship': 0.3042535580850136},
'mmAP': 0.3911842063674406,
'0.5': {'roundabout': 0.6422598076912881, 'baseball-diamond': 0.6484967025788626, 'swimming-pool': 0.544313314911686, 'helicopter': 0.45170116916451697, 'ground-track-field': 0.5368440177649815, 'basketball-court': 0.5775599813741871, 'storage-tank': 0.8192065993263431, 'large-vehicle': 0.6302270061303114, 'small-vehicle': 0.6440849311638136, 'mAP': 0.6515208634123666, 'soccer-ball-field': 0.6676082230060035, 'harbor': 0.6266462479561381, 'plane': 0.8960098115915338, 'bridge': 0.37363580515970096, 'tennis-court': 0.891465236437895, 'ship': 0.8227540969282348},
'0.65': {'roundabout': 0.5180425677564794, 'baseball-diamond': 0.5245568874311204, 'swimming-pool': 0.38272168522044336, 'helicopter': 0.3448772155458963, 'ground-track-field': 0.5060681005637672, 'basketball-court': 0.5485980593478017, 'storage-tank': 0.7726328443507657, 'large-vehicle': 0.5050278241827602, 'small-vehicle': 0.5564416466669729, 'mAP': 0.556107336784344, 'soccer-ball-field': 0.6011672943653977, 'harbor': 0.36039060249818533, 'plane': 0.8885506601099703, 'bridge': 0.21821284624591858, 'tennis-court': 0.8910745096861368, 'ship': 0.7232473077935442},
'0.95': {'roundabout': 0.0, 'baseball-diamond': 0.006993006993006993, 'swimming-pool': 0.0, 'helicopter': 0.0, 'ground-track-field': 0.0, 'basketball-court': 0.0, 'storage-tank': 0.022727272727272728, 'large-vehicle': 0.0005411255411255411, 'small-vehicle': 0.0008045052292839903, 'mAP': 0.013501678263490986, 'soccer-ball-field': 0.012987012987012986, 'harbor': 0.0, 'plane': 0.003134796238244514, 'bridge': 0.0, 'tennis-court': 0.15466405356301732, 'ship': 0.0006734006734006734},
'0.55': {'roundabout': 0.6420649898459735, 'baseball-diamond': 0.6183154782708276, 'swimming-pool': 0.5198382571301288, 'helicopter': 0.44864412140897136, 'ground-track-field': 0.5332751996286547, 'basketball-court': 0.5775599813741871, 'storage-tank': 0.8055197347924181, 'large-vehicle': 0.5915287152140464, 'small-vehicle': 0.631233465283797, 'mAP': 0.6309545724101561, 'soccer-ball-field': 0.6566350227586047, 'harbor': 0.5303328697660692, 'plane': 0.8953186062284073, 'bridge': 0.33135587596292976, 'tennis-court': 0.8914370047361784, 'ship': 0.7912592637511461},
'0.9': {'roundabout': 0.09090909090909091, 'baseball-diamond': 0.03636363636363637, 'swimming-pool': 0.0008658008658008659, 'helicopter': 0.0012804097311139564, 'ground-track-field': 0.045454545454545456, 'basketball-court': 0.06734006734006734, 'storage-tank': 0.06893675303339926, 'large-vehicle': 0.004999159805074777, 'small-vehicle': 0.0404040404040404, 'mAP': 0.07708073310701025, 'soccer-ball-field': 0.16558441558441558, 'harbor': 0.0019793581224374383, 'plane': 0.0948471959156224, 'bridge': 0.004329004329004329, 'tennis-court': 0.5251932169049558, 'ship': 0.0077243018419489015},
'0.7': {'roundabout': 0.4839091066119396, 'baseball-diamond': 0.4223450953926336, 'swimming-pool': 0.24104529432118618, 'helicopter': 0.262598391655095, 'ground-track-field': 0.43799218302870024, 'basketball-court': 0.5371561951901982, 'storage-tank': 0.7160371906173102, 'large-vehicle': 0.4222642587644076, 'small-vehicle': 0.45342090304643146, 'mAP': 0.4842854160769299, 'soccer-ball-field': 0.5666586326843455, 'harbor': 0.24169288358945845, 'plane': 0.8003639228589434, 'bridge': 0.1639642918023705, 'tennis-court': 0.8905465545762308, 'ship': 0.6242863370146984},
'0.85': {'roundabout': 0.17488636363636365, 'baseball-diamond': 0.10349650349650351, 'swimming-pool': 0.00686106346483705, 'helicopter': 0.08373205741626794, 'ground-track-field': 0.10497835497835498, 'basketball-court': 0.2371657754010695, 'storage-tank': 0.2680411552283259, 'large-vehicle': 0.04786784079353014, 'small-vehicle': 0.10503330661433428, 'mAP': 0.1886599059789572, 'soccer-ball-field': 0.27662878787878786, 'harbor': 0.09090909090909091, 'plane': 0.40150109234662407, 'bridge': 0.022727272727272728, 'tennis-court': 0.7610197405936645, 'ship': 0.14505018419933155},
'0.75': {'roundabout': 0.3941059616488605, 'baseball-diamond': 0.30595713084562326, 'swimming-pool': 0.1503781374447408, 'helicopter': 0.17069341412001113, 'ground-track-field': 0.3899806141246988, 'basketball-court': 0.49780332292041757, 'storage-tank': 0.619527459864304, 'large-vehicle': 0.32541154570851444, 'small-vehicle': 0.3129437868800671, 'mAP': 0.40585005106114813, 'soccer-ball-field': 0.496271032776619, 'harbor': 0.16663041026940834, 'plane': 0.7777892503191356, 'bridge': 0.09759325713273082, 'tennis-court': 0.8833821546167296, 'ship': 0.4992832872453607},
'0.6': {'roundabout': 0.595336001515304, 'baseball-diamond': 0.5806983217034197, 'swimming-pool': 0.4705005825336498, 'helicopter': 0.4037572432010673, 'ground-track-field': 0.5153968512659525, 'basketball-court': 0.5638120475754269, 'storage-tank': 0.7853180410454478, 'large-vehicle': 0.5471913324590196, 'small-vehicle': 0.6082839939072276, 'mAP': 0.5979144860106378, 'soccer-ball-field': 0.6522167227732261, 'harbor': 0.42692173077126255, 'plane': 0.893792088997363, 'bridge': 0.2926216687293154, 'tennis-court': 0.8914370047361784, 'ship': 0.741433658945707}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_2x_20210127'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'

# ---------------------------------------- System
ROOT_PATH = os.path.abspath('../../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 20673 * 2

SUMMARY_PATH = os.path.join(ROOT_PATH, 'output/summary')
TEST_SAVE_PATH = os.path.join(ROOT_PATH, 'tools/test_result')

pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_R_DIR = os.path.join(ROOT_PATH, 'output/evaluate_result_pickle/')

# ------------------------------------------ Train and Test
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True
ADD_BOX_IN_TENSORBOARD = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ALPHA = 1.0
BETA = 1.0

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 1e-3
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Dataset
DATASET_NAME = 'DOTATrain'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 15

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
FPN_CHANNEL = 256
NUM_SUBNET_CONV = 4
FPN_MODE = 'fpn'

# --------------------------------------------- Anchor
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
USE_ANGLE_COND = False
ANGLE_RANGE = 90  # or 180

# -------------------------------------------- Head
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

