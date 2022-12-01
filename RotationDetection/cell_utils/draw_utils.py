import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def _adjust_boxes(boxes, h, w):
    boxes[:,:,0] *= w
    boxes[:,:,1] *= h
    boxes = np.int0(boxes)
    return boxes

def draw_ann(path_img, ann):
    
    img = cv2.imread(path_img)
    h,w,_ = img.shape
    boxes = _adjust_boxes(np.copy(ann), h, w)
    
    cv2.drawContours(img, boxes, -1, (0,255,0), 2)
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    
def get_centers(path_img, ann, resize=None):
    img = cv2.imread(path_img)
    if resize:
        img = cv2.resize(img, resize)
    h,w,_ = img.shape
    
    boxes = _adjust_boxes(np.copy(ann), h, w)
    
    circles = np.zeros(shape=img.shape[:2], dtype='uint8')
    for box in boxes:
        cx = int(np.mean(box[:,0]))
        cy = int(np.mean(box[:,1]))
        
        circles[cy,cx] = 255
        
    return circles

def get_shapes(path_img, ann, resize=None):
    img = cv2.imread(path_img)
    if resize:
        img = cv2.resize(img, resize)
    h,w,_ = img.shape
    
    boxes = _adjust_boxes(np.copy(ann), h, w)
    shapes = [cv2.minAreaRect(box) for box in boxes]
        
    return shapes

def draw_heatmap(path_img, ann):
    
    img = cv2.imread(path_img)
    h,w,_ = img.shape
    
    boxes = _adjust_boxes(np.copy(ann), h, w)
    
    draw = np.zeros(img.shape, dtype='float32')
    for box in boxes:
        
        cx = int(np.mean(box[:,0]))
        cy = int(np.mean(box[:,1]))
        
        _draw = np.zeros_like(draw)
        cv2.circle(_draw, (cx,cy), 5, (0,0,255), -1)
        
        rect = cv2.minAreaRect(box)
        cv2.ellipse(_draw, rect, (0,255,0), -1)
        
        draw += _draw
    
    draw[...,1] = draw[...,1]/(np.max(draw[...,1])+1e-6)*255
    draw[...,2] = draw[...,2]/(np.max(draw[...,2])+1e-6)*255
    
    return img, np.uint8(draw)