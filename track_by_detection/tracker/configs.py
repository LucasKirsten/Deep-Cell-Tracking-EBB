# -*- coding: utf-8 -*-
"""
Define global configuration values

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import os

#%% values to be set by the user

DEBUG = True

DATASET  = 'Fluo-N2DH-GOWT1'
LINEAGE  = '01'
FRAME_SHAPE = (1024,1024)
DETECTOR = 'r2cnn'
DIVIDE_PREDS = False
FROM_CROPS = False

path_imgs = f'../../../{DATASET}/{LINEAGE}'
path_dets = f'../{DATASET}_results'
path_normal_dets  = os.path.join(path_dets, 'det_normal_cell.txt')
path_mitoses_dets = os.path.join(path_dets, 'det_mitoses.txt')

# value to filter individual detections based on minimun score
# for normal detections
NORMAL_SCORE_TH = 0.5
# for mitoses detection
MIT_SCORE_TH = 0.5

# threshold to use in order to join detections on NMS algorithm
NMS_TH = 0.5

#%% values to compute the tracking algorithm

# thresholds for the tracker
TRACK_SCORE_TH = 0.9

# distance between frames to consider transposition
# higher values means that higher gaps allows to join detections
LINK_TH = 3

# distance between frames to consider mitoses
# higher values means that allows higher gaps to consider a mitoses event
MIT_TH = 3

# distance between cell centers in pixels to consider joining
# higher values means that cells far appart can be joined in tracklets
CENTER_TH = 0.1*(FRAME_SHAPE[0]**2+FRAME_SHAPE[1]**2)**(1/2)

# distance between cell centers in pixels for mitoses
CENTER_MIT_TH = 0.1*(FRAME_SHAPE[0]**2+FRAME_SHAPE[1]**2)**(1/2)

# values to adjust the probabilities distributions
# higher values means larger probabilites
LINK_FACTOR = 25
MIT_FACTOR  = 100

#%% values that were calculated previously

# alpha values for the networks (based on their P50 values)
# add values for each detector (normal, mitoses)

ALPHAS = {
    'glioblastoma': {'':{
        'dcl':      (0.7181,0.6335),
        'csl':      (0.7242,0.5398),
        'rsdet':    (0.7433,0.6360),
        'retinanet':(0.7525,0.5647),
        'r3det':    (0.7515,0.6675),
        'r3detdcl': (0.7550,0.6263),
        'r2cnn':    (0.7733,0.6965)
        }},
    
    'Fluo-N2DH-GOWT1': {
        '01':{
            'r2cnn': (0.8993,0.8993)},
        '02':{
            'r2cnn': (0.8644,0.8644)},
        },
        
     'Fluo-N2DL-HeLa': {
        '01':{
            'r2cnn_x2': (0.7930,0.7930),
            'r2cnn_x2_segtra': (0.8604,0.8604)},
        '02':{
            'r2cnn_x2': (0.7899,0.7899)},
        },
    
    'PhC-C2DH-U373': {
        '01':{
            'r2cnn': (0.7673,0.7673)},
        '02':{
            'r2cnn': (0.6867,0.6867)}
        },
    }

# map values according to the alpha values above
ALPHA_NORMAL = ALPHAS[DATASET][LINEAGE][DETECTOR][0]
ALPHA_MITOSE = ALPHAS[DATASET][LINEAGE][DETECTOR][1]
