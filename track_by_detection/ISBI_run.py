# -*- coding: utf-8 -*-
"""
Example of tracking by detection

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)

TBD:
    - Self learning
    - Solve hyphotesis matrix in batches of N tracklets/frames
    - Remove bad frames
    - Adjust frame position if camera moved
"""

import os
import cv2
import sys
from time import time

from tracker import *

use_watershed = sys.argv[-1]=='True'

#%% read all detections
init = time()

# get sorted frames by name
frame_imgs = [file.split('.')[0] \
               for file in os.listdir(path_imgs) \
               if file.endswith('.tif')]

# get detections
detections = read_detections(path_normal_dets, path_mitoses_dets, \
                             frame_imgs, from_crops=FROM_CROPS)

#%% split detections into frames
frames = get_frames_from_detections(detections, frame_imgs)
Nf = len(frames)
del detections

#%% apply NMS on frames detections
nms_frames = apply_NMS(frames)

#%% get trackelts
tracklets = get_tracklets(nms_frames)

#%% solve tracklets
final_tracklets = solve_tracklets(tracklets, Nf)

if DEBUG: print('Elapsed time: ', time()-init)

#%% evaluate using ISBI
ISBI_evaluate(path_imgs+'_RES', final_tracklets, Nf, \
              evaluate=False, use_watershed=use_watershed)

