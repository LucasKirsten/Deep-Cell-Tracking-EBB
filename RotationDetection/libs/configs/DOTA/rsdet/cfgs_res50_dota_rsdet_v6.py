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
SAVE_WEIGHTS_INTE = 20673 * 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTATrain'

# model
# backbone
pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0 / 5.0

VERSION = 'RetinaNet_DOTA_2x_RSDet_5p_20210129'

"""
RSDet-5p  20epoch
FLOPs: 860451115;    Trainable params: 33002916
{'0.6': {'ship': 0.699516961917484, 'bridge': 0.2868883283346906, 'roundabout': 0.5477989696955544, 'basketball-court': 0.5633916659860057, 'soccer-ball-field': 0.6464876154223445, 'baseball-diamond': 0.6171071618440039, 'ground-track-field': 0.5543768471575958, 'storage-tank': 0.7698611840630336, 'swimming-pool': 0.36850118226596873, 'mAP': 0.5583773545337656, 'tennis-court': 0.9022095376792877, 'plane': 0.8897505869841165, 'large-vehicle': 0.41461869618024033, 'harbor': 0.32964552731855135, 'small-vehicle': 0.5149415943928791, 'helicopter': 0.27056445876472573},
'0.7': {'ship': 0.46627026086873774, 'bridge': 0.10440702306731015, 'roundabout': 0.42620546108321045, 'basketball-court': 0.5020036599090156, 'soccer-ball-field': 0.5992149285040892, 'baseball-diamond': 0.4372293470192242, 'ground-track-field': 0.4120860768535057, 'storage-tank': 0.6592882845950556, 'swimming-pool': 0.17016069206139284, 'mAP': 0.4232072817288323, 'tennis-court': 0.879109467828116, 'plane': 0.7818531235836765, 'large-vehicle': 0.2472568983146774, 'harbor': 0.13742315914768322, 'small-vehicle': 0.35665543142766026, 'helicopter': 0.16894541166913016},
'0.9': {'ship': 0.003405389398874741, 'bridge': 0.0014204545454545455, 'roundabout': 0.09090909090909091, 'basketball-court': 0.045454545454545456, 'soccer-ball-field': 0.08093841642228738, 'baseball-diamond': 0.018181818181818184, 'ground-track-field': 0.09090909090909091, 'storage-tank': 0.08046587369017275, 'swimming-pool': 0.0, 'mAP': 0.05451702565781812, 'tennis-court': 0.3166978297608341, 'plane': 0.07492548820504141, 'large-vehicle': 0.002668564312399929, 'harbor': 0.009090909090909092, 'small-vehicle': 0.002687913986752424, 'helicopter': 0.0},
'0.95': {'ship': 0.00017316017316017316, 'bridge': 0.0, 'roundabout': 0.0, 'basketball-court': 0.007575757575757575, 'soccer-ball-field': 0.03636363636363637, 'baseball-diamond': 0.0, 'ground-track-field': 0.0, 'storage-tank': 0.0037878787878787876, 'swimming-pool': 0.0, 'mAP': 0.008576015866718039, 'tennis-court': 0.05783926218708828, 'plane': 0.022727272727272728, 'large-vehicle': 0.00017327018597666628, 'harbor': 0.0, 'small-vehicle': 0.0, 'helicopter': 0.0},
'0.65': {'ship': 0.594133089366667, 'bridge': 0.22010614225394826, 'roundabout': 0.49343957134851246, 'basketball-court': 0.546268852751328, 'soccer-ball-field': 0.6251323302760831, 'baseball-diamond': 0.5413387364515183, 'ground-track-field': 0.4692097001172344, 'storage-tank': 0.7333605118398658, 'swimming-pool': 0.26041359424159416, 'mAP': 0.49971426534337104, 'tennis-court': 0.8955415114886078, 'plane': 0.8808528855145068, 'large-vehicle': 0.32420845099375045, 'harbor': 0.2413610327760997, 'small-vehicle': 0.4460067004291466, 'helicopter': 0.2243408703017026},
'0.85': {'ship': 0.03251258893096826, 'bridge': 0.01652892561983471, 'roundabout': 0.11900826446280992, 'basketball-court': 0.1712121212121212, 'soccer-ball-field': 0.1479309221244705, 'baseball-diamond': 0.054819277108433734, 'ground-track-field': 0.11122994652406418, 'storage-tank': 0.23697934427585918, 'swimming-pool': 0.002457002457002457, 'mAP': 0.11791183926965973, 'tennis-court': 0.542680474929432, 'plane': 0.26076480620341536, 'large-vehicle': 0.010480747322852585, 'harbor': 0.02727272727272727, 'small-vehicle': 0.025709531509995547, 'helicopter': 0.009090909090909092},
'0.5': {'ship': 0.741919571642551, 'bridge': 0.38158480633069536, 'roundabout': 0.6278845462240081, 'basketball-court': 0.5976141745280927, 'soccer-ball-field': 0.691817194858993, 'baseball-diamond': 0.7006471929688352, 'ground-track-field': 0.6432269948185928, 'storage-tank': 0.7837478618462681, 'swimming-pool': 0.5182435129854706, 'mAP': 0.6349813334303456, 'tennis-court': 0.9041377852841251, 'plane': 0.893020989321137, 'large-vehicle': 0.5310523702416563, 'harbor': 0.5269792754496666, 'small-vehicle': 0.591816502685401, 'helicopter': 0.39102722226969044},
'0.75': {'ship': 0.3217299860370234, 'bridge': 0.04213430602319491, 'roundabout': 0.38208704673082106, 'basketball-court': 0.45841942148760334, 'soccer-ball-field': 0.4996416788026889, 'baseball-diamond': 0.32688497780913983, 'ground-track-field': 0.3333948416786146, 'storage-tank': 0.5492979221173278, 'swimming-pool': 0.07955233459181138, 'mAP': 0.33320212939945987, 'tennis-court': 0.7922120597213534, 'plane': 0.6574826388360392, 'large-vehicle': 0.14015995052558566, 'harbor': 0.07842639593908629, 'small-vehicle': 0.21660838069160754, 'helicopter': 0.12},
'0.8': {'ship': 0.1459312931955865, 'bridge': 0.0303030303030303, 'roundabout': 0.25931854429462087, 'basketball-court': 0.364484126984127, 'soccer-ball-field': 0.34814814814814815, 'baseball-diamond': 0.1396853146853147, 'ground-track-field': 0.22927688562198606, 'storage-tank': 0.41371609199359416, 'swimming-pool': 0.01127554615926709, 'mAP': 0.22902050138227967, 'tennis-court': 0.7624230667239196, 'plane': 0.509340938339888, 'large-vehicle': 0.06772376766899144, 'harbor': 0.03636363636363637, 'small-vehicle': 0.10433011726507231, 'helicopter': 0.012987012987012986},
'mmAP': 0.34606150192423074,
'0.55': {'ship': 0.728140976909981, 'bridge': 0.3222835384510777, 'roundabout': 0.5978858905450949, 'basketball-court': 0.5753636516422018, 'soccer-ball-field': 0.6755547374692368, 'baseball-diamond': 0.6435326426903206, 'ground-track-field': 0.622953241406792, 'storage-tank': 0.7790308798775247, 'swimming-pool': 0.4554317901435609, 'mAP': 0.6011072726300578, 'tennis-court': 0.9036408075588203, 'plane': 0.8911095129071771, 'large-vehicle': 0.49055503019083035, 'harbor': 0.4258999537608187, 'small-vehicle': 0.5523264209985959, 'helicopter': 0.35290001489883566}}
"""



