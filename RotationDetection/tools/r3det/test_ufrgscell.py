# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append("../../")

#from track_by_detection.tracker.classes import Detection, Frame
#from track_by_detection.tracker.preprocessing import apply_NMS

from libs.models.detectors.r3det import build_whole_network
from tools.test_ufrgscell_base import TestUFRGSCELL
from libs.configs import cfgs
from libs.val_libs.voc_eval_r import EVAL


class TestUFRGSCELLGWD(TestUFRGSCELL):

    def eval(self):
        r3det = build_whole_network.DetectionNetworkR3Det(cfgs=self.cfgs,
                                                      is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=r3det,
                                          image_ext=self.args.image_ext)
        #print([len(box) for box in all_boxes_r])
        #
        ## apply NMS
        #frames = []
        #for fr in all_boxes_r:
        #    detections = Frame()
        #    
        #    for box in fr:
        #        mit,score,cx,cy,w,h,ang = box
        #        
        #        det = Detection(None,score=score,cx=cx,cy=cy,w=w,h=h,ang=ang,mit=mit-1)
        #        detections.append(det)
        #        
        #    frames.append(detections)
        #
        #nms_boxes = apply_NMS(frames)
        #print([len(box) for box in nms_boxes])
        #all_boxes_r = []
        #for fr in nms_boxes:
        #    arr = fr.get_values()[:,[-1,0,1,2,3,4,5]]
        #    arr[:,0] += 1
        #    all_boxes_r.append(np.array(arr))
        #
        imgs = os.listdir(self.args.img_dir)
        
        ## draw detections
        #print('Drawing detections...')
        #for path,boxes in zip(imgs, all_boxes_r):
        #    image = cv2.imread(os.path.join(self.args.img_dir, path))
        #    
        #    draw = np.copy(image)
        #    for box in boxes:
        #        cx,cy,w,h,a = box[2:]
        #        box = cv2.boxPoints(((cx,cy),(w,h),a))
        #        box = np.int0(box)
        #
        #        draw = cv2.drawContours(draw, [box], -1, (0,255,0), 2)
        #        
        #    path_save = '/workdir/msc/RotationDetection/output/detections'
        #    path_save = os.path.join(path_save, path)
        #    cv2.imwrite(path_save, draw)
        
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]
        
        #with open('/workdir/msc/RotationDetection/output/results_pickle/r3det.pkl', 'wb') as handle:
        #    pickle.dump({'all_boxes':all_boxes_r, 'imgs':imgs}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':

    tester = TestUFRGSCELLGWD(cfgs)
    tester.eval()
