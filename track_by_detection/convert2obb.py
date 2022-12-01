# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:27:46 2022

@author: kirstenl
"""

import os
import sys
import cv2
import shutil
import argparse
import numpy as np
import tifffile as tiff
from numpy import linalg as LA
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--lineage", type=str)
parser.add_argument("--path_root", type=str)
parser.add_argument("--augment", type=bool, default=False)
parser.add_argument("--augment_size", type=float)
parser.add_argument("--consider_mitoses", type=bool, default=False)
parser.add_argument("--copy_tra2seg", type=bool, default=False)

args = parser.parse_args(sys.argv[1:])
print(args)

dataset   = args.dataset
lineage   = args.lineage
path_root = args.path_root
augment   = args.augment
augment_size = args.augment_size
if augment:
    assert augment_size is not None, 'Should provide augment_size if augment==True'
consider_mitoses = args.consider_mitoses
copy_tra2seg = args.copy_tra2seg

#%%

def norm(x):
    return np.uint8((x-np.min(x))/(np.max(x)-np.min(x))*255)

path_images = sorted(glob(path_root+f'/{dataset}/{lineage}_GT/SEG/*.tif'))
path_save_ann = path_root+f'/{dataset}/{lineage}/annotations/dota_format'
path_res = path_root+f'/{dataset}/{lineage}_RES'
tracklets = open(path_root+f'/{dataset}/{lineage}_GT/TRA/man_track.txt').read()
tracklets = [list(map(int, track.split(' '))) for track in tracklets.split('\n')[:-1]]
    
#%%

if augment:
    def _resize_image(path):
        image = cv2.imread(path, -1)
        image = norm(image)
        # augment image
        image = cv2.resize(image, (int(image.shape[1]*augment_size),int(image.shape[0]*augment_size)))
        
        cv2.imwrite(path.replace('.tif', '.jpg'), image)
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        pbar = tqdm(glob(path_root+f'/{dataset}/{lineage}/images/*.tif'))
        pbar.set_description('Resizing images...')
        _ = parallel(delayed(_resize_image)(path) for path in pbar)

if copy_tra2seg:
    def _copy_seg2tra(path_seg):
        path_tra = path_seg.replace('SEG/man_seg', 'TRA/man_track')
        seg = cv2.imread(path_seg, -1)
        tra = cv2.imread(path_tra, -1)
        
        kernel = np.ones((3,3),np.uint8)
        tra = cv2.dilate(tra, kernel)
        
        seg = seg | tra
        
        tiff.imsave(path_seg, seg)
        
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        pbar = tqdm(path_images)
        pbar.set_description('Copying labels from tracking to segmentation...')
        _ = parallel(delayed(_copy_seg2tra)(path_seg) for path_seg in pbar)
    
#%% iterate over tracklets to find mitoses

mitoses = {} # cell idx : frame position
if consider_mitoses:
    for track in tracklets[::-1]:
        if track[0] in mitoses:
            mitoses[track[0]] = track[-2]
        elif track[-1]==0:
            continue
        else:
            mitoses[track[-1]] = -1

#%%
os.makedirs(path_res, exist_ok=True)
os.makedirs(path_save_ann, exist_ok=True)

def _adjust_segmentation(frame, path):
    
    if not path.endswith('.tif'):
        return
    
    img_name = os.path.split(path)[-1]
    file = open(os.path.join(path_save_ann, 't'+img_name[-7:-4]+'.txt'), 'w')
    file.write('imagesource:ISBI\ngsd:0\n')
    
    image = cv2.imread(path, -1)
    # augment image
    if augment:
        image = cv2.resize(image, (int(image.shape[1]*augment_size),int(image.shape[0]*augment_size)) ,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, image)
        
        # augment for tracking
        path_tra = path.replace('SEG/man_seg', 'TRA/man_track')
        tra = cv2.imread(path_tra, -1)
        tra = cv2.resize(tra, (image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path_tra, tra)
    
    vals = np.unique(image)[1:]
    
    draw = np.zeros_like(image)
    for sval in vals:
        y,x = np.where(image==sval)
        if len(x)<2 or len(y)<2:
            continue
        
        points = np.array(list(zip(x,y)))
        (cx,cy), (w,h), ang = cv2.minAreaRect(points)
        
        ellipse = ((cx,cy),(w,h),ang)
        
        draw = cv2.ellipse(draw, ellipse, (int(sval),), -1)
        
        box = cv2.boxPoints(ellipse)
        box = np.int0(box).reshape(-1)
        box[::2]  = np.clip(box[::2], 0, image.shape[1])
        box[1::2] = np.clip(box[1::2], 0, image.shape[0])
        
        cell_class = 'normal_cell'
        if (sval in mitoses) and (frame == mitoses[sval]):
            cell_class = 'mitoses'
        
        file.write(' '.join(map(lambda x:str(x), box))+f' {cell_class} 0\n')
    
    cv2.imwrite(os.path.join(path_res, 'mask'+img_name[-7:]), draw)
    file.close()

with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
    _ = parallel(delayed(_adjust_segmentation)(frame, path) \
                 for frame,path in tqdm(enumerate(path_images), total=len(path_images)))