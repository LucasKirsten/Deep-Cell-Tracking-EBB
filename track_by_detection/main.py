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
from time import time

from tracker import *

#%% read all detections
init = time()

# get sorted frames by name
frame_imgs = [file.split('.')[0] for file in os.listdir(path_imgs)]

# get detections
detections = read_detections(f'{path_dets}/det_normal_cell.txt', \
                             f'{path_dets}/det_mitoses.txt', frame_imgs,\
                             from_crops=True)

#%% split detections into frames
frames = get_frames_from_detections(detections, frame_imgs)
Nf = len(frames)
#del detections

#%% apply NMS on frames detections
nms_frames = apply_NMS(frames)

#%%
# draw_detections(nms_frames, path_imgs, img_format='.tif', plot=True)

#%% get trackelts
tracklets = get_tracklets(nms_frames)

#%% solve tracklets
final_tracklets = solve_tracklets(tracklets, Nf)

if DEBUG: print('Elapsed time: ', time()-init)

#%% write results in csv file

# write_results(f'./{DATASET}_results.csv', tracklets, nms_frames)

#%% evaluate using ISBI
ISBI_evaluate(f'./frames/{DATASET}/{LINEAGE}_RES', final_tracklets, Nf)

#%% evaluate predictions

# if os.path.exists(path_gt):
#     annotations = read_annotations(path_gt)
#     print(MOTA_evaluate(annotations, final_tracklets, Nf, 'center'))

#%% draw trackings
# frame_imgs = draw_tracklets(final_tracklets, nms_frames, path_imgs, \
#                             img_format='.tif', save_frames=True)

# if DEBUG: print('Elapsed time with drawing: ', time()-init)

