# -*- coding: utf-8 -*-
"""
Function to evaluate the final tracklet results.

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import os
import cv2
import numpy as np
import motmetrics as mm
import tifffile as tiff
from numba import njit
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .classes import Frame
from .func_utils import helinger_dist, intersection_over_union

@njit(parallel=True, cache=True)
def _build_hd_matrix(frm0, frm1):
    # compute the distances using the Helinger distance
    costs = np.zeros((len(frm0), len(frm1)))
    for j in range(costs.shape[0]):
        for k in range(costs.shape[1]):
            cx0,cy0,w0,h0,ang0,a0,b0,c0 = frm0[j][1:-1] # remove score and mit
            cx1,cy1,w1,h1,ang1,a1,b1,c1 = frm1[k][1:-1]
            # compute helinger distance and IoU
            hd = helinger_dist(cx0,cy0,a0,b0,c0, cx1,cy1,a1,b1,c1)
            iou = intersection_over_union(cx0,cy0,w0,h0, cx1,cy1,w1,h1)
            costs[k,j] = hd if iou>0 else np.nan
    return costs

def MOTA_evaluate(true_tracklets:list, pred_tracklets:list, num_frames:int,\
             dist_method:str='hd') -> str:
    '''
    Evaluate the tracklets results using the MOTA metric.

    Parameters
    ----------
    true_tracklets : list
        List of true tracklets.
    pred_tracklets : list
        List of predicted tracklets.
    num_frames : int
        Total number of frames.
    dist_method : str, optional
        Method to compute the distance between true and predicted detections on each frame.\
            Should be either center, iou or hd. The default is 'hd'.

    Returns
    -------
    strsummary : str
        Summary of the results.

    '''
    
    assert dist_method in ('center', 'iou', 'hd'), \
        'Distance method should be either center, iou or hd!'
    
    # convert tracklets from predictions to frames
    pred_frames = [Frame() for _ in range(num_frames+1)]
    for i,track in enumerate(pred_tracklets):
        fr_idx = track.start
        for d_idx,det in enumerate(track):
            det.idx = i
            pred_frames[fr_idx+d_idx].append(det)
    
    # convert tracklets from annotations to frames
    true_frames = [Frame() for _ in range(num_frames+1)]
    for i,track in enumerate(true_tracklets):
        fr_idx = track.start
        for d_idx,det in enumerate(track):
            det.idx = i
            true_frames[fr_idx+d_idx].append(det)
    
    # define accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    # add all detections to accumulator
    pbar = zip(pred_frames, true_frames)
    if DEBUG:
        pbar = tqdm(pbar, total=len(pred_frames))
        pbar.set_description('Computing prediction and annotation assignments')
    for pred,true in pbar:
        
        idx_pred = pred.get_idxs()
        idx_true = true.get_idxs()
        
        # compute distance between predicted and true detections using 
        # the chose metric
        if len(true)<1 or len(pred)<1:
            continue
        
        if dist_method=='center':
            c_pred = pred.get_centers()
            c_true = true.get_centers()
            dists = mm.distances.norm2squared_matrix(c_true, c_pred)
            
        elif dist_method=='iou':
            c_pred = pred.get_iou_values()[:,:-1]
            c_true = true.get_iou_values()[:,:-1]
            dists = mm.distances.iou_matrix(c_true, c_pred)
            
        elif dist_method=='hd':
            c_pred = pred.get_values()
            c_true = true.get_values()
            dists = _build_hd_matrix(c_true, c_pred)
        
        acc.update(idx_true, idx_pred, dists)
        
    # make evaluation
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]],
        metrics=mm.metrics.motchallenge_metrics,
        names=['full', 'part'], generate_overall=True)
    
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    return strsummary

#%% function to draw and write tracklets evaluation for ISB challenge

def ISBI_evaluate(path_save:str, tracklets:list, Nf:int, \
                  evaluate:bool=True, use_watershed:bool=False) -> None:
    
    os.makedirs(path_save, exist_ok=True)
    frame_dets = [[] for _ in range(Nf)]
    
    file = open(os.path.join(path_save, 'res_track.txt'), 'w')
    
    pbar = tracklets
    if DEBUG:
        pbar = tqdm(pbar, total=len(tracklets))
        pbar.set_description('Writing tracklets')
    for track in pbar:
        
        parent = 0 if track.parent is None else track.parent.idx
        file.write(f'{track.idx} {track.start} {track.end} {parent}\n')
        
        start = track.start
        for di,det in enumerate(track):
            if start+di>Nf:
                continue
            frame_dets[start+di].append(det)
        
    file.close()
    
    pbar = enumerate(frame_dets)
    if DEBUG:
        pbar = tqdm(pbar, total=Nf)
        pbar.set_description('Drawing frames')
    def _draw_frames(i,frame):
        draw = np.zeros(FRAME_SHAPE, dtype='uint16')
        dets = sorted(frame, key=lambda x:-x.area)
        for det in dets:
            ellipse = ((det.cx,det.cy),(det.w,det.h),det.a)
            cv2.ellipse(draw, ellipse, det.idx, -1)
        
        if use_watershed:
            zeros = np.zeros((*FRAME_SHAPE,3), dtype='uint8')
            draw = cv2.watershed(zeros,np.int32(draw))
            draw[draw==-1] = 0

        tiff.imsave(os.path.join(path_save, f'mask{i:03d}.tif'), np.uint16(draw))
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_draw_frames)(i,frame) for i,frame in pbar)
        
    if evaluate:
        os.system(f'cd evaluation/Linux && ./DETMeasure /workdir/{DATASET} {LINEAGE} 3')
        os.system(f'cd evaluation/Linux && ./SEGMeasure /workdir/{DATASET} {LINEAGE} 3')
        os.system(f'cd evaluation/Linux && ./TRAMeasure /workdir/{DATASET} {LINEAGE} 3')
            
    
    
    
    
    
    
    
    
