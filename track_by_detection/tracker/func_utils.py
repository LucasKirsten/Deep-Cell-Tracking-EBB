# -*- coding: utf-8 -*-
"""
Functions used by other modules.

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import numpy as np
from numpy import linalg as LA
import pandas as pd
from numba import njit
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

from .configs import *

#%% intersection over union

@njit
def intersection_over_union(cxA,cyA,wA,hA, cxB,cyB,wB,hB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(cxA-wA/2, cxB-wB/2)
	yA = max(cyA-hA/2, cyB-hB/2)
	xB = min(cxA+wA/2, cxB+wB/2)
	yB = min(cyA+hA/2, cyB+hB/2)

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (wA + 1) * (hA + 1)
	boxBArea = (wB + 1) * (hB + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
    
	return iou

#%% helinger distance
EPS = 1e-3

@njit
def helinger_dist(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2, sw=1., cw=1.):
    
    B1 = (a1+a2)*(y1-y2)**2. + (b1+b2)*(x1-x2)**2.
    B1 += 2.*(c1+c2)*(x2-x1)*(y1-y2)
    B1 /= (a1+a2)*(b1+b2)-(c1+c2)**2+EPS
    B1 /= 4.
    
    B2 = (a1+a2)*(b1+b2)-(c1+c2)**2.
    B2 /= 4.*np.sqrt((a1*b1-c1**2.)*(a2*b2-c2**2.))+EPS
    B2 = 1./2.*np.log(B2)
    
    Bd = cw*B1+sw*B2
    Bc = np.exp(-Bd)
    
    Hd = np.sqrt(1.-Bc+EPS)
    
    if Hd>1+2*EPS:
        raise Exception('Value larger than 1 to Hd')
    elif Hd<0 or np.isnan(Hd):
        raise Exception('Value smaller than 0 or nan to Hd')
    
    return Hd if Hd<1 else 1.0

@njit
def get_hd(cx,cy,w,h,angle):
    # get Helinger distance values
    angle *= np.pi/180.
    
    al = w**2./12.
    bl = h**2./12.
    
    a = al*np.cos(angle)**2.+bl*np.sin(angle)**2.
    b = al*np.sin(angle)**2.+bl*np.cos(angle)**2.
    c = 1./2.*(al-bl)*np.sin(2.*angle)
    return cx,cy,a,b,c

@njit
def get_from_hd(cx,cy,a,b,c):
    # get standard values from Helinger distance ones
    
    corr = np.array([[a,c],[c,b]])
    val,vec = LA.eig(corr)
    w = np.sqrt(np.abs(val[0])*12.)
    h = np.sqrt(np.abs(val[1])*12.)
    ang = vec[1][0]*180./np.pi
    
    return cx,cy,w,h,ang

@njit
def center_distances(x1,y1, x2,y2):
    # computer distance between center points
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

#%% functions to calculate probabilities

def PTP(Xk, alpha):
    return (Xk.score()-MIT_SCORE_TH)*alpha

def PFP(Xk, alpha):
    return (1-alpha)*(1-Xk.score()+MIT_SCORE_TH)**len(Xk)

def Plink(Xj, Xi, cnt_dist):
    measure = cnt_dist*(Xi.start-Xj.end)
    return np.exp(-measure/LINK_FACTOR)

def Pmit(cnt_dist, d_mit):
    measure = cnt_dist*d_mit
    return np.exp(-measure/MIT_FACTOR)


    
    