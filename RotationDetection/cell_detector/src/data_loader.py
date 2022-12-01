import sys
sys.path.append('../')

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import configs as cfgs
from dataloader.dataset.image_augmentation import ImageAugmentation

image_preprocess = ImageAugmentation(cfgs)
AUTOTUNE = tf.data.experimental.AUTOTUNE

def _read_data(serialized_example):
    # read TF records data format
    features = tf.parse_single_example(
            serialized=serialized_example,
            features={
                'img_name': tf.FixedLenFeature([], tf.string),
                'img_height': tf.FixedLenFeature([], tf.int64),
                'img_width': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
                'num_objects': tf.FixedLenFeature([], tf.int64)
            }
        )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)
    
    img = tf.reshape(img, shape=[img_height, img_width, 3])
    
    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])
    
    num_objects = tf.cast(features['num_objects'], tf.int32)
    return tf.cast(img, 'float32'), gtboxes_and_label

def _augment_data(img, gtboxes_and_label):
    if cfgs.RGB2GRAY:
        img = image_preprocess.random_rgb2gray(img_tensor=img, gtboxes_and_label=gtboxes_and_label)

    if cfgs.IMG_ROTATE:
        img, gtboxes_and_label = image_preprocess.random_rotate_img(img_tensor=img,
                                                                    gtboxes_and_label=gtboxes_and_label)
        
    img, gtboxes_and_label, img_h, img_w = image_preprocess.short_side_resize(img_tensor=img,
                                                                              gtboxes_and_label=gtboxes_and_label,
                                                                              target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                                              length_limitation=cfgs.IMG_MAX_LENGTH)

    if cfgs.HORIZONTAL_FLIP:
        img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img,
                                                                         gtboxes_and_label=gtboxes_and_label)
    if cfgs.VERTICAL_FLIP:
        img, gtboxes_and_label = image_preprocess.random_flip_up_down(img_tensor=img,
                                                                      gtboxes_and_label=gtboxes_and_label)
        
    return tf.cast(img, 'float32'), gtboxes_and_label

def _get_gt(class_centers, regres_centers, reduce=16):

    # placeholder for placing the centers
    shape = class_centers.shape[0]//reduce, class_centers.shape[1]//reduce
    gt_regression = np.zeros(shape=(shape[0], shape[1], cfgs.REGRESSIONS))
    gt_classification = np.zeros(shape=(shape[0], shape[1], cfgs.CLASS_NUM))

    # first map all unique center points and save the conflits
    conflits = []
    for gh,h in enumerate(range(reduce//2, class_centers.shape[0]-reduce//2, reduce)):
        for gw,w in enumerate(range(reduce//2, class_centers.shape[1]-reduce//2, reduce)):
            # get the label patch
            classes = class_centers[h-reduce//2:h+reduce//2, w-reduce//2:w+reduce//2]

            # get the number of center points
            nbr_centers = np.count_nonzero(classes)

            if nbr_centers==1:
                # get regression values
                regression = regres_centers[h-reduce//2:h+reduce//2, w-reduce//2:w+reduce//2]
                regression = np.sum(regression, axis=(0,1))

                # adjust cx and cy
                regression[0] -= gw*reduce
                regression[1] -= gh*reduce
                
                gt_regression[gh,gw] = regression
                
                # get the class value
                gt_classification[gh,gw,np.sum(classes)-1] = 1

            if nbr_centers>1:
                conflits.append([[gh,gw], [h,w]])
    
    # iterate over patches with conflits
    for (gh,gw), (h,w) in conflits:
        # get the conflit points in the patch
        classes = class_centers[h-reduce//2:h+reduce//2, w-reduce//2:w+reduce//2]
        regression = regres_centers[h-reduce//2:h+reduce//2, w-reduce//2:w+reduce//2]

        # get point of label
        cnt_pts = np.where(classes>0)
        cnt_pts = [[cnt_pts[0][i],cnt_pts[1][i]] for i in range(len(cnt_pts))]

        # calculate the distance to the borders
        dleft, dright, dtop, dbottom = [],[],[],[]
        for pt in cnt_pts:
            dleft.append(pt[1])
            dright.append(gt_regression.shape[0]-pt[0])
            dtop.append(pt[0])
            dbottom.append(gt_regression.shape[1]-pt[1])

        # calculate the distance to the center and sort related to how close they are to the center
        cnt_dist = [np.square((p[0]-reduce/2.)**2. + (p[1]-reduce/2.)**2.) for p in cnt_pts]
        arg_pts = np.argsort(cnt_dist, kind='mergesort')

        # get the regression values
        regression = [regression[pt[0],pt[1]] for p in cnt_pts]
        classes = [classes[pt[0],pt[1]] for p in cnt_pts]

        # now fill the gaps related to the conflits
        for i in arg_pts:
            # verify the best displacement
            extreme = np.array([dleft[i], dright[i], dtop[i], dbottom[i]])
            extreme = np.argsort(extreme, kind='mergesort')

            # try to fit in the sorted best direction
            found = False
            ww,hh = 0,0
            for k in range(gt_regression.shape[0]):
                for disp in extreme:
                    if disp==0: # left
                        if np.sum(gt_regression[gh, gw-k])==0 and gw-k>0:
                            ww = -k
                            found = True
                    elif disp==1: # right
                        if gw+k<gt_regression.shape[1] and np.sum(gt_regression[gh, gw+k])==0:
                            ww = k
                            found = True
                    elif disp==2: # top
                        if np.sum(gt_regression[gh-k, gw])==0 and gh-k>0:
                            hh = -k
                            found = True
                    else: # bottom
                        if gh+k<gt_regression.shape[0] and np.sum(gt_regression[gh+k, gw])==0:
                            hh = k
                            found = True
                    if found:
                        # adjust center values and store to annotations
                        regression[i][0] -= (gw+ww)*reduce
                        regression[i][1] -= (gh+hh)*reduce
                        
                        gt_regression[gh+hh, gw+ww] = regression[i]
                        gt_classification[gh+hh, gw+ww, classes[i]-1] = 1
                        break
                if found:
                    break
    
    # labels normalization
    gt_regression[...,0] /= 32 # cx
    gt_regression[...,1] /= 32 # cy
    gt_regression[...,2] /= cfgs.IMG_MAX_LENGTH # w
    gt_regression[...,3] /= cfgs.IMG_MAX_LENGTH # h
    gt_regression[...,4] /= -90 # angle
    
    return gt_regression, gt_classification

def _adjust_data(img, labels):
    
    # norm image
    img = (tf.cast(img, 'float32')-cfgs.PIXEL_MEAN)/cfgs.PIXEL_STD
    
    # add annotations to the center values of each class
    centers_regres = np.zeros(shape=(*img.shape[:2],cfgs.REGRESSIONS), dtype='int32')
    centers_class = np.zeros(shape=img.shape[:2], dtype='int32')
    for lb in labels:
        # get boxes values
        box = lb[:-1].reshape(-1,2)
        (cx,cy), (w,h), angle= cv2.minAreaRect(box)

        # if center already has values, raise error
        if np.sum(centers_class[int(cy),int(cx)])>1:
            raise Exception()

        centers_regres[int(cy),int(cx)] = [cx,cy,w,h,angle]
        centers_class[int(cy),int(cx)] = lb[-1]+1
    
    centers_regres, centers_class = _get_gt(centers_class, centers_regres)
    
    return img, centers_regres.astype('float32'), centers_class.astype('float32')

@tf.function(input_signature=[tf.TensorSpec((None,None,None), tf.float32), tf.TensorSpec((None,9), tf.int32)])
def _tf_adjust_data(img, label):
    return tf.py_func(_adjust_data, [img, label], [tf.float32, tf.float32, tf.float32])

# adjust to feed network
def _adjust2net(img, regress, classes):
    return img, {'regression':regress, 'classification':classes}

def data_loader():
        return tf.data.TFRecordDataset(tf.io.gfile.glob(cfgs.PATH_TFRECORDS+'*.tfrecord')) \
            .shuffle(256) \
            .map(_read_data, num_parallel_calls=AUTOTUNE) \
            .map(_augment_data, num_parallel_calls=AUTOTUNE) \
            .map(_tf_adjust_data, num_parallel_calls=AUTOTUNE) \
            .map(_adjust2net, num_parallel_calls=AUTOTUNE) \
            .batch(cfgs.BATCH_SIZE) \
            .prefetch(AUTOTUNE) \
            .apply(tf.data.experimental.ignore_errors()) \
            .repeat() 

def view_data(img, classes, regression, reduce=16):
    
    draw = np.copy(img)
    colors = [(255,0,0), (0,255,0)]
    
    # desnormalization of regressions
    regression[...,0] *= 32 # cx
    regression[...,1] *= 32 # cy
    regression[...,2] *= cfgs.IMG_MAX_LENGTH # w
    regression[...,3] *= cfgs.IMG_MAX_LENGTH # h
    regression[...,4] *= -90 # angle
    
    for cl in range(cfgs.CLASS_NUM):
        cnt = classes[...,cl]
        
        for gh in range(cnt.shape[0]):
            for gw in range(cnt.shape[1]):
                if cnt[gh,gw]==0:
                    continue
                
                reg = regression[gh,gw]
                reg[0] += gw*reduce
                reg[1] += gh*reduce
                
                rect = ((reg[0],reg[1]), (reg[2],reg[3]), reg[4])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(draw, [box], -1, colors[cl], 2)
    
    plt.figure(figsize=(10,10))
    plt.imshow(draw)