# -*- coding: utf-8 -*-
"""
Auxiliar Classes for the tracking algorithm

@author: Lucas N. Kirsten (lnkirsten@inf.ufrgs.br)
"""

import cv2
import numpy as np
from shapely.geometry import Polygon

from .func_utils import get_hd, get_from_hd, center_distances as cdist

class Detection():
    def __init__(self,frame:str,score:float,cx:float,cy:float,\
                 w:float=None,h:float=None,ang:float=None,\
                 a:float=None,b:float=None,c:float=None,mit:int=0,\
                 idx:int=-1, convert:bool=True):
        '''
        Define a detection.

        Parameters
        ----------
        frame : str
            Name of the detection frame.
        score : float
            Detection score.
        cx : float
            x center position.
        cy : float
            y center position.
        w : float, optional
            Width size. The default is None.
        h : float, optional
            Height size. The default is None.
        ang : float, optional
            Angle value. The default is None.
        a : float, optional
            a value from the covariance matrix. The default is None.
        b : float, optional
            b value from the covariance matrix. The default is None.
        c : float, optional
            c value from the covariance matrix. The default is None.
        mit : int, optional
            If detection is mitose (mit=1) or not (mit=0). The default is 0.
        idx : int, optional
            Index value of detection (detection with same idx belongs to same tracklet). The default is -1.
        convert : bool, optional
            If to convert values of one representation to another (e.g., standard (cx,cy,w,h,ang) to (cx,cy,a,b,c)). The default is True.
        '''
        
        self.frame = frame
        self.score = float(score)
        self.mit = int(mit)
        self.idx = idx
        
        self.cx = float(cx)
        self.cy = float(cy)
        self.area = None
        
        if convert:
            if (a is None) or (b is None) or (c is None):
                self.w, self.h, self.ang = float(w), float(h), float(ang)
                self.cx,self.cy,self.a,self.b,self.c = \
                    get_hd(self.cx,self.cy,self.w,self.h,self.ang)
                assert self.a*self.b-self.c**2>=0, 'Error computing HD values'
                    
            elif (w is None) or (h is None) or (ang is None):
                self.a, self.b, self.c = float(a), float(b), float(c)
                assert self.a*self.b-self.c**2>=0, 'Error with assigned HD values'
                self.cx,self.cy,self.w,self.h,self.ang = \
                    get_from_hd(self.cx,self.cy,self.a,self.b,self.c)
            
            else:
                self.w,self.h,self.ang = float(w), float(h), float(ang)
                self.a,self.b,self.c = float(a),float(b),float(c)
        
        else:
            self.w,self.h,self.ang,self.a,self.b,self.c = 0,0,0,0,0,0
            
        assert self.a*self.b-self.c**2>=0, 'Error with HD values'
                
        if self.w is not None and self.h is not None:
            box = cv2.boxPoints(((self.cx,self.cy),(self.w,self.h),self.ang))
            box = np.int0(box)
            self.box = Polygon(box)
            self.area = self.box.area
                
    def __str__(self):
        return str(self.__dict__)
    
    def get_values(self):
        return self.cx,self.cy,self.a,self.b,self.c
    
    def iou(self, det) -> float:
        '''
        Get the intersection value percentage between two detections.

        Parameters
        ----------
        det : Detection
            Detection to compute the intersection with.

        Returns
        -------
        float
            Intersection value between the two detections.

        '''
        return self.box.intersection(det.box).area / self.box.union(det.box).area
        
class Tracklet():
    def __init__(self, detections, start:int):
        '''
        Define a Tracklet.

        Parameters
        ----------
        detections : list or Detection
            List of detections, or single detection to compose Tracklet.
        start : int
            Start value relative to the frames.
        '''
        
        if type(detections)==list:
            self.detections = detections
            self.sum_score = np.sum([d.score for d in detections])
        else:
            self.detections = [detections]
            self.sum_score = detections.score
        
        # compute tracklet speed and mean area
        self.distance, self.sum_area = 0., 0.
        if len(self.detections)>1:
            self.distance = sum([cdist(self[i].cx,self[i-1].cx,self[i].cy,self[i-1].cy) \
                                 for i in range(1,len(self.detections))])
            self.sum_area = sum([det.area for det in self])
        
        self.start = start
        self.size = len(self.detections)
        self.end = start + self.size - 1
        self.parent = None
        self.idx = None
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.detections[i]
    
    def __add(self, k):
        self.size += k
        self.end += k
    
    def append(self, x):
        x.idx = self.detections[-1].idx
        self.detections.append(x)
        self.sum_score += x.score
        self.distance += cdist(x.cx,self[-1].cx,x.cy,self[-2].cy)
        self.sum_area += x.area
        self.__add(1)
        
    def set_idx(self, i):
        self.idx = i
        for det in self:
            det.idx = i
        
    def score(self):
        return self.sum_score/self.size
    
    def slice_score(self, start, end):
        return np.mean([d.score for d in self[start:end]])
    
    def speed(self):
        return self.distance/(self.size-1)
    
    def area(self):
        return self.sum_area/self.size
    
    # join two tracklets (e.g., for translation hyphotesis)
    def join(self, tracklet):
        '''
        Join one tracklet to another.

        Parameters
        ----------
        tracklet : Tracklet
            The tracklet to be joined with.
        '''
        
        assert tracklet.start>self.end, f'Trying to join non consecutive tracklets with ids {tracklet.idx},{self.idx}!'
        
        # if there are gaps between tracklets, linearly fill them
        if tracklet.start - self.end > 1:
            dx = tracklet.start - self.end - 1
            d0 = self[-1]
            df = tracklet[0]
            for i in range(dx):
                
                parms = {'cx':None,'cy':None,'w':None,'h':None,'ang':None}
                for p in parms.keys():
                    parms[p] = d0.__dict__[p] + (i+1)*(df.__dict__[p]-d0.__dict__[p])/dx
                
                score = (self[-1-i].score + df.score)/2.
                parms.update({'frame':None,'score':score,'mit':0})
                
                self.append(Detection(**parms))
            
        # add all detections to the tracklet
        for det in tracklet:
            det.idx = self.detections[-1].idx
            self.append(det)
        
        # adjust parenting
        if (self.parent is None) and (tracklet.parent is not None):
            self.parent = tracklet.parent
            
    def split_mitoses(self, d_mit:int=0) -> list:
        '''
        Split the tracklet where a mitoses event occurs.
        
        Parameters
        ----------
        d_mit : int, optional
            Distance from the mitoses event to the tracklet start. The default value is 0.

        Returns
        -------
        list
            The two resulting tracklets from splitting.
        '''
        
        t1 = Tracklet(self[:d_mit], self.start)
        t2 = Tracklet(self[d_mit:], self.start + d_mit)
        
        return t1, t2
    
    def split(self, d_split:int):
        '''
        TBD

        Parameters
        ----------
        d_split : int
            DESCRIPTION.

        Returns
        -------
        track : TYPE
            DESCRIPTION.

        '''
        
        track = Tracklet(self[:d_split], self.start)
        track.set_idx(self.idx)
        return track
        
        
class Frame(list):
    def __init__(self, values:list=[], name:str=None):
        '''
        Define a Frame.

        Parameters
        ----------
        values : list, optional
            A list of values to the initially define the Frame list. The default is [].
        name : str, optional
            Frame image name. The default is None.
        '''
        super(Frame, self).__init__(values)
        self.name = name
    
    def get_values(self):
        '''
        Returns a numpy array containing all score,cx,cy,w,h,ang,a,b,c,d,mit values of detections in the Tracklet.
        '''
        return np.array([[d.score,d.cx,d.cy,d.w,d.h,d.ang,d.a,d.b,d.c,d.mit]\
                         for d in self])
    
    def get_centers(self):
        '''
        Returns a numpy array containing all cx,cy values of detections in the Tracklet.
        '''
        return np.array([[d.cx,d.cy] for d in self])
    
    def get_iou_values(self):
        '''
        Returns a numpy array containing all cx,cy,w,h,ang values of detections in the Tracklet.
        '''
        return np.array([[d.cx,d.cy,d.w,d.h,d.ang] for d in self])
    
    def get_hd_values(self):
        '''
        Returns a numpy array containing all cx,cy,a,b,c values of detections in the Tracklet.
        '''
        return np.array([[d.cx,d.cy,d.a,d.b,d.c] for d in self])
    
    def get_idxs(self):
        '''
        Returns a numpy array containing all idx values of detections in the Tracklet.
        '''
        return np.array([d.idx for d in self])
    
    
    
    
    