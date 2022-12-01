# -*- coding: utf-8 -*-
"""
Auxiliar functions from drawing results.

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *

def draw_detections(frames:list, path_imgs:list, img_format:str='.png', \
                    plot:bool=False, save_video:bool=False) -> list:
    '''
    Draw detections on the Frames.

    Parameters
    ----------
    frames : list
        List of frames.
    path_imgs : list
        Path to all image frames.
    img_format : str, optional
        The extension of images in the image path. The default is '.png'.
    plot : bool, optional
        If to plot the results. The default is False.
    save_video : bool, optional
        If to save the results in a video format. The default is False.

    Returns
    -------
    list
        List of draw frames results.

    '''
    
    pbar = frames
    if DEBUG:
        pbar = tqdm(pbar)
        pbar.set_description('Reading frames')
    def _draw_frame(frm):
        # read images and draw the detectors detections
        img_name = frm.name
        img_name = os.path.join(path_imgs, img_name+img_format)
        img = cv2.imread(img_name, -1)
        img = np.uint8(255*np.float32(img-img.min())/(img.max()-img.min()))
        if len(img.shape)<3:
            img = cv2.merge([img]*3)
        draw = np.copy(img)
        
        # iterate over detections to draw in frame
        for det in frm:
            cx,cy,w,h,a,mit = det.cx,det.cy,det.w,det.h,det.a,det.mit
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            color = (0,0,255) if int(mit)==0 else (0,255,0)
            draw = cv2.drawContours(draw, [box], -1, color, 2)
        
        return draw
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        frame_imgs = parallel(delayed(_draw_frame)(frm) for frm in pbar)
        
    if plot:
        # if to plot the final frames results
        for img in frame_imgs:
            cv2.imshow('', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if save_video:
        # if to save a video with the final frame results
        h,w,c = frame_imgs[0].shape
        out = cv2.VideoWriter(f'./{DATASET}_{DETECTOR}_{len(frames)}.avi',\
                              cv2.VideoWriter_fourcc(*'XVID'), 10.0, (w,h))
        for img in frame_imgs:
            out.write(img)
        out.release()
        
    return frame_imgs

def draw_tracklets(tracklets:list, frames:list, path_imgs:list, img_format:str='.png', \
                   plot:bool=False, save_video:bool=False, save_frames:bool=False):
    '''
    Draw tracklets on the Frames.

    Parameters
    ----------
    tracklets : list
        List of tracklets.
    frames : list
        List of frames.
    path_imgs : list
        Path to all image frames.
    img_format : str, optional
        The extension of images in the image path. The default is '.png'.
    plot : bool, optional
        If to plot the results. The default is False.
    save_video : bool, optional
        If to save the results in a video format. The default is False.

    Returns
    -------
    list
        List of draw frames results.

    '''
    
    frame_imgs = draw_detections(frames, path_imgs, img_format, plot=False)
    
    total_detections = len(tracklets)
    colors = np.linspace(10,240,total_detections+1,dtype='uint8')[1:]
    np.random.shuffle(colors)
    
    pbar = enumerate(tracklets)
    if DEBUG:
        pbar = tqdm(pbar, total=len(tracklets))
        pbar.set_description('Drawing tracklets')
    def _draw_track(ti, track):
        nonlocal frame_imgs, colors
        
        start = track.start
        det_id = str(track.idx)
        if track.parent is not None:
            det_id += f'.{track.parent.idx}'
        
        for di,det in enumerate(track):
            if start+di>=len(frames):
                continue
     
            cx,cy,w,h,a = det.cx,det.cy,det.w,det.h,det.a
            box = cv2.boxPoints(((cx,cy),(w,h),a))
            box = np.int0(box)
            
            color = (int(colors[ti]), int(255-colors[ti]), 255)
            frame_imgs[start+di] = cv2.drawContours(frame_imgs[start+di], [box], -1, color, 2)
            frame_imgs[start+di] = cv2.putText(frame_imgs[start+di], det_id, (int(cx),int(cy)), \
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
            #frame_imgs[start+di] = cv2.circle(frame_imgs[start+di], (int(cx),int(cy)), 2, color, 2)
    
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        _ = parallel(delayed(_draw_track)(ti,track) for ti,track in pbar)

    if plot:
        for img in frame_imgs:
            cv2.imshow('', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if save_video:
        h,w,c = frame_imgs[0].shape
        out = cv2.VideoWriter(f'./{DATASET}_{DETECTOR}_{len(frames)}.avi',\
                              cv2.VideoWriter_fourcc(*'XVID'), 10.0, (w,h))
        for img in frame_imgs:
            out.write(img)
        out.release()
        
    if save_frames:
        path_save = f'./results/{DATASET}_{LINEAGE}_{DETECTOR}'
        os.makedirs(path_save, exist_ok=True)
        for i,img in enumerate(frame_imgs):
            cv2.imwrite(f'{path_save}/t{i}.jpg', img)
        
    return frame_imgs