# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from importlib import import_module
sys.path.append("../")

from utils import tools
from libs.label_name_dict.label_dict import LabelMap
from libs.utils.draw_box_in_img import DrawBox
from libs.utils.coordinate_convert import forward_convert, backward_convert
from libs.utils import nms_rotate
from libs.utils.rotate_polygon_nms import rotate_gpu_nms

import configs as cfgs


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test HRSC2016')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/data/dataset/HRSC2016/HRSC2016/Test/AllImages', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.jpg', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='/data/dataset/HRSC2016/HRSC2016/Test/xmls', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)
    parser.add_argument('--draw_imgs', '-s', default=False,
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def resize2ratio(img):
    h,w,c = img.shape
    hh = h%2**5; ww = w%2**5
    
    img = cv2.copyMakeBorder(img, 0, hh, 0, ww, cv2.BORDER_CONSTANT)
    
    return img

def get_detections(classes, regression, reduce=16):
    
    # desnormalization of regressions
    regression[...,0] *= 32 # cx
    regression[...,1] *= 32 # cy
    regression[...,2] *= cfgs.IMG_MAX_LENGTH # w
    regression[...,3] *= cfgs.IMG_MAX_LENGTH # h
    regression[...,4] *= -90 # angle
    
    detection_boxes, detection_scores, detection_category = [],[],[]
    for cl in range(cfgs.CLASS_NUM):
        cnt = classes[...,cl]
        
        for gh in range(cnt.shape[0]):
            for gw in range(cnt.shape[1]):
                if cnt[gh,gw]<0.1:
                    continue
                
                reg = regression[gh,gw]
                reg[0] += gw*reduce
                reg[1] += gh*reduce
                
                rect = ((reg[0],reg[1]), (reg[2],reg[3]), reg[4])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                detection_boxes.append(box)
                detection_scores.append(cnt[gh,gw])
                detection_category.append(cl+1)
    
    return detection_boxes, detection_scores, detection_category


class TestUFRGSCELL(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        label_map = LabelMap(cfgs)
        self.name_label_map, self.label_name_map = label_map.name2label(), label_map.label2name()

    def eval_with_plac(self, img_dir, image_ext):

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        
        det_net = import_module(f'{cfgs.EXPERIMENT_NAME}.model').Model()
        det_net.load_weights(f'results/{cfgs.EXPERIMENT_NAME}.h5')

        all_boxes_r = []
        imgs = os.listdir(img_dir)
        pbar = tqdm(imgs)
        for a_img_name in pbar:

            a_img_name = a_img_name.split(image_ext)[0]
            try:
                raw_img = cv2.imread(os.path.join(img_dir, a_img_name + image_ext))
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
            except Exception as e:
                print('\n' + str(e))
                print(os.path.join(img_dir, a_img_name + image_ext))
                continue

            det_boxes_r_all, det_scores_r_all, det_category_r_all = [], [], []
            
            img_resize = np.copy(raw_img)

            img_batch = (img_resize-self.cfgs.PIXEL_MEAN)/self.cfgs.PIXEL_STD
            img_batch = resize2ratio(img_batch)
            img_batch = tf.expand_dims(img_batch, axis=0)
            
            classes, regress = det_net.predict(img_batch)
            det_boxes_r_all, det_scores_r_all, det_category_r_all = get_detections(classes[0], regress[0])
            det_boxes_r_all = np.array(det_boxes_r_all)
            det_scores_r_all = np.array(det_scores_r_all)
            det_category_r_all = np.array(det_category_r_all)

            if det_boxes_r_all.shape[0] == 0:
                continue
            
            box_res_rotate_, label_res_rotate_, score_res_rotate_ = [],[],[]
            if det_scores_r_all.shape[0] != 0:
                for sub_class in range(1, self.cfgs.CLASS_NUM + 1):
                    index = np.where(det_category_r_all == sub_class)[0]
                    if len(index) == 0:
                        continue
                    tmp_boxes_r = det_boxes_r_all[index]
                    tmp_label_r = det_category_r_all[index]
                    tmp_score_r = det_scores_r_all[index]
                    inx = np.arange(0, tmp_score_r.shape[0])

                    box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                    score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                    label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            if len(box_res_rotate_) == 0:
                all_boxes_r.append(np.array([]))
                continue

            det_boxes_r_ = np.array(box_res_rotate_)
            det_scores_r_ = np.array(score_res_rotate_)
            det_category_r_ = np.array(label_res_rotate_)

            if self.args.draw_imgs:
                detected_indices = det_scores_r_ >= self.cfgs.VIS_SCORE
                detected_scores = det_scores_r_[detected_indices]
                detected_boxes = det_boxes_r_[detected_indices]
                detected_categories = det_category_r_[detected_indices]

                detected_boxes = backward_convert(detected_boxes, False)

                drawer = DrawBox(self.cfgs)

                det_detections_r = drawer.draw_boxes_with_label_and_scores(raw_img[:, :, ::-1],
                                                                           boxes=detected_boxes,
                                                                           labels=detected_categories,
                                                                           scores=detected_scores,
                                                                           method=1,
                                                                           in_graph=True)

                save_dir = os.path.join('test_ufrgscell', self.cfgs.VERSION, 'ufrgscell_img_vis')
                tools.makedirs(save_dir)

                cv2.imwrite(save_dir + '/{}.jpg'.format(a_img_name),
                            det_detections_r[:, :, ::-1])

            det_boxes_r_ = backward_convert(det_boxes_r_, False)

            x_c, y_c, w, h, theta = det_boxes_r_[:, 0], det_boxes_r_[:, 1], det_boxes_r_[:, 2], \
                                    det_boxes_r_[:, 3], det_boxes_r_[:, 4]

            boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
            dets_r = np.hstack((det_category_r_.reshape(-1, 1),
                                det_scores_r_.reshape(-1, 1),
                                boxes_r))
            all_boxes_r.append(dets_r)

            pbar.set_description("Eval image %s" % a_img_name)

        return all_boxes_r


