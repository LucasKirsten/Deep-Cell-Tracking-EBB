# -*- coding: utf-8 -*-
"""
Functions to handle frames/tracklets.

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import product
from tqdm import tqdm
from numba import njit
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .func_utils import *
from .classes import *

#%% get the frames from detections

def get_frames_from_detections(detections:list, frame_names:list) -> list:
    '''
    Return a list of frames from a list of detections.

    Parameters
    ----------
    detections : list
        List of detections.
    frame_names : list
        List of the frame image names.

    Returns
    -------
    list
        Detections aggregated into frames.

    '''
    
    # merge detections into frames
    frames = {name:Frame(name=name) for name in frame_names}
    
    pbar = detections
    if DEBUG:
        pbar = tqdm(pbar)
        pbar.set_description('Loading frames: ')
    for det in pbar:
        frames[det.frame].append(det)
        
    # sort the frames by name
    frames = [frames[name] for name in sorted(list(frames.keys()))]
        
    return frames

#%% get frames from tracklets

def get_frames_from_tracklets(tracklets:list, frame_names:list) -> list:
    
    # merge detections into frames
    frames = [Frame(name=name) for name in sorted(frame_names)]
    
    pbar = tracklets
    if DEBUG:
        pbar = tqdm(pbar)
        pbar.set_description('Loading frames: ')
    for track in pbar:
        start = track.start
        for i,det in enumerate(track):
            frames[start+i].append(det)
        
    return frames

#%% iterate over frames to apply hungarian algorithm and get tracklets

@njit(parallel=True, cache=True)
def _build_costs(frm0, frm1):
    # build cost matrix for the Hungarian algorithm
    costs = np.zeros((len(frm0), len(frm1)))
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            s0,cx0,cy0,w0,h0,ang0,a0,b0,c0 = frm0[j][:-1] # remove score and mit
            s1,cx1,cy1,w1,h1,ang1,a1,b1,c1 = frm1[k][:-1]
            # compute helinger distance
            costs[j,k] = helinger_dist(cx0,cy0,a0,b0,c0, cx1,cy1,a1,b1,c1, .5)
    return costs

def get_tracklets(frames:list) -> list:
    '''
    Get a list of tracklets from a list of frames.

    Parameters
    ----------
    frames : list
        List of frames.

    Returns
    -------
    list
        Aggregated tracklets from frames detections.

    '''
    
    # add a indexing value for the first frame detections
    for n,det in enumerate(frames[0]):
        det.idx = n
    
    # add a -1 index value for all other frames detections
    for fr in frames[1:]:
        for det in fr:
            det.idx = -1
    
    # initialize trackelts
    tracklets = [Tracklet(det,0) for det in frames[0]]
    
    # set of indexes
    ids = set(range(len(tracklets)))
    
    pbar = range(len(frames)-1)
    if DEBUG:
        pbar = tqdm(pbar)
        pbar.set_description('Getting tracklets: ')
    for i in pbar:
        
        # take consecutive frames
        frm0 = frames[i]
        frm1 = frames[i+1]
        
        # hungarian algorithm
        costs = _build_costs(frm0.get_values(), frm1.get_values())
        row_ind, col_ind = linear_sum_assignment(costs)
        
        # map the detected objects to its pairs
        for row,col in zip(row_ind, col_ind):
            if costs[row][col]<0.5:
                frm1[col].idx = int(frm0[row].idx)
                tracklets[frm1[col].idx].append(frm1[col])
            else:
                frm1[col].idx = int(max(ids) + 1)
                ids.add(max(ids) + 1)
                tracklets.append(Tracklet([frm1[col]], i+1))
        
        # verify if any detection remained with idx==-1
        for det in frm1:
            if int(det.idx) == -1:
                det.idx = int(max(ids) + 1)
                ids.add(max(ids) + 1)
                tracklets.append(Tracklet([det], i+1))
    
    # sort tracklets based on the first frame they appear
    tracklets = sorted(tracklets, key=lambda x:x.start)
    
    # set idx values
    for i,track in enumerate(tracklets):
        track.set_idx(i+1)
                
    return tracklets











