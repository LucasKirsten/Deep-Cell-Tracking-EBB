# -*- coding: utf-8 -*-
"""
Functions to read data.

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import os
import cv2
import numpy as np
from numpy import linalg as LA
import pandas as pd
from numba import njit
from tqdm import tqdm
from glob import glob
import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *
from .classes import *

#%% functions to read detections

def _read(path_dets, frame_imgs, threshold, mit, from_crops):
    # auxiliar function to read the detections
    
    try:
        dets = pd.read_csv(path_dets, header=None, sep=' ')
    except:
        return []
    dets = dets.sort_values(by=0)
    if not from_crops:
        dets = dets.loc[dets.iloc[:,0].isin(frame_imgs)]
    
    # filter detections by score
    dets = dets.loc[dets.iloc[:,1]>threshold]
    
    if from_crops:
        detections = []
        for _,det in dets.iterrows():
            frame,score,cx,cy,w,h,ang = det
            frame,stridey,stridex = frame.split('_')
            if not frame in frame_imgs:
                continue
            cx += float(stridex)
            cy += float(stridey)
            
            if DIVIDE_PREDS:
                cx /= 2
                cy /= 2
                w /= 2
                h /= 2
            
            detections.append(Detection(frame,score,cx,cy,w,h,ang,mit=mit))
    else:
        detections = [Detection(*det,mit=mit) for _,det in dets.iterrows()]
    
    return detections

def read_detections(path_normals:str, path_mitoses:str,\
                    frame_imgs:list, from_crops:bool=False) -> list:
    '''
    Read txt files with normal and mitoses detections.

    Parameters
    ----------
    path_normals : str
        Path to the file containing the normal detections.
    path_mitoses : str
        Path to the file containing the mitoses detections.
    frame_imgs : list
        List of frames image names.
    from_crops: bool
        Wheater the detections are from crops of the full size images. The default value is False.

    Returns
    -------
    list
        List of detections.

    '''
    
    # sort frame images by name
    frame_imgs = sorted(frame_imgs)
    
    # open normal detections
    if DEBUG: print('Reading normal detections...')
    detections = _read(path_normals, frame_imgs, NORMAL_SCORE_TH, 0, from_crops)

    # open mitoses detections
    if DEBUG: print('Reading mitoses detections...')
    detections.extend(_read(path_mitoses, frame_imgs, MIT_SCORE_TH, 1, from_crops))

    # sort detections by name
    return detections #sorted(detections, key=lambda x:x.frame)
    
#%% functions to read labels

@njit
def _is_continuos(frames_id):
    # verify if the frames detections are continuos
    for i in range(len(frames_id)-1):
        if frames_id[i]+1!=frames_id[i+1]:
            return 0
    return 1

def _read_annotations_csv(path_gt):
    df = pd.read_csv(path_gt)
    
    assert ('cx' in df) and ('cy' in df) and ('frame' in df), \
           'The annotation should contain at least columns frame,cx and cy!'
    
    # fill missing columns
    if not ('w' in df):
        df['w'] = [0]*len(df['cx'])
    if not ('h' in df):
        df['h'] = [0]*len(df['cx'])
    
    # group by cell track and remove non continous ones
    groups = [x for _, x in df.groupby(['cell'])]
    groups = [g for g in groups if _is_continuos(np.array(g)[:,1])]
    
    def _get_tracklets(i,group):
        # auxiliar function to define tracklets from detections
        detections = [Detection(r['frame'],1,r['cx'],r['cy'],r['w'],r['h'],\
                                idx=i,convert=False) \
                      for _,r in group.iterrows()]
        return Tracklet(detections, start=detections[0].frame-1)
        
    pbar = enumerate(groups)
    if DEBUG:
        pbar = tqdm(pbar, total=len(groups))
        pbar.set_description('Reading annotations')
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        tracklets = parallel(delayed(_get_tracklets)(i,group) \
                               for i,group in pbar)
        
    return tracklets

def _read_annotations_tif(path_gt:str, ext:str='.tif'):
    # read annotations from tif files
    
    path_gt = glob(os.path.join(path_gt, '*'+ext))
    path_gt = sorted(path_gt)
    
    tracklets = {}
    
    # iterate over the image paths
    pbar = enumerate(path_gt)
    if DEBUG:
        pbar = tqdm(pbar, total=len(path_gt))
        pbar.set_description('Reading annotations')
    for i,path in pbar:
        
        # read the ground truth image
        frame = os.path.split(path)[-1].split('.')[-0]
        seg = cv2.imread(path, -1)
        
        for val in np.unique(seg):
            if val==0:
                continue
            
            # get the rectangle region of the gt cell
            y,x = np.where(seg==val)
            if len(x)<2 or len(y)<2:
                continue
            
            points = np.array(list(zip(x,y)))
            (cx,cy), (w,h), ang = cv2.minAreaRect(points)
            
            # cx = np.mean(x)
            # cy = np.mean(y)
            
            # corr = np.cov(x,y)
            # vecval,vec = LA.eig(corr)
            # w = np.sqrt(np.abs(vecval[0])*12.)
            # h = np.sqrt(np.abs(vecval[1])*12.)
            # ang = vec[1][0]*180./np.pi
            
            # define the detection
            detection = Detection(frame,1,cx,cy,w,h,ang)           
            
            # add to the stored tracklets
            if val not in tracklets:
                tracklets[val] = Tracklet(detection, i)
            else:
                tracklets[val].append(detection)
                
    return list(tracklets.values())

def read_annotations(path_gt:str, ext:str='.tif') -> list:
    '''
    Read the annotation data from a folder containing tif images, or a csv file.

    Parameters
    ----------
    path_gt : str
        Path to the annotations. Should be either a folder containing tif images or a csv file.
    ext : str, optional
        Extension of the ground truth images, if path_gt is a folder. The default is '.tif'.

    Raises
    ------
    Exception
        If path_gt is not a csv file or a folder.

    Returns
    -------
    list
        List of ground truth tracklets.

    '''
    
    # verify how to read the ground truth data
    if path_gt.endswith('.csv'):
        return _read_annotations_csv(path_gt)
    elif os.path.isdir(path_gt):
        return _read_annotations_tif(path_gt, ext)
    else:
        raise Exception('path_gt should reference to a csv file or a folder \
                        containing tif images!')

#%% functions to write results

def write_results(path_save:str, tracklets:list, frames:Frame) -> None:
    
    data = {'frame':[], 'id':[], 'cx':[], 'cy':[], 'w':[], 'h':[], 'angle':[]}
    
    for track in tracklets:
        start = track.start
        for i,det in enumerate(track):
            data['frame'].append(frames[start+i].name)
            data['id'].append(det.idx)
            data['cx'].append(det.cx)
            data['cy'].append(det.cy)
            data['w'].append(det.w)
            data['h'].append(det.h)
            data['angle'].append(det.ang)
            
    pd.DataFrame(data).to_csv(path_save, index=False)
    

