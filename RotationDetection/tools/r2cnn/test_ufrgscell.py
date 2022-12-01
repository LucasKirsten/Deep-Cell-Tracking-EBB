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

from libs.models.detectors.r2cnn import build_whole_network
from tools.test_ufrgscell_base import TestUFRGSCELL
from libs.configs import cfgs
from libs.val_libs.voc_eval_r import EVAL


class TestUFRGSCELLGWD(TestUFRGSCELL):

    def eval(self):
        r2cnn = build_whole_network.DetectionNetworkR2CNN(cfgs=self.cfgs,
                                                          is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=r2cnn,
                                          image_ext=self.args.image_ext)
        #print([len(box) for box in all_boxes_r])
        imgs = [p for p in os.listdir(self.args.img_dir) if p.endswith(self.args.image_ext)]
        
        ## merge detections with r3det results
        #with open('/workdir/msc/RotationDetection/output/results_pickle/r3det.pkl', 'rb') as handle:
        #    r3det_results = pickle.load(handle)
        #    
        #for i in range(len(all_boxes_r)):
        #    assert imgs[i]==r3det_results['imgs'][i]
        #    all_boxes_r[i] = np.concatenate([all_boxes_r[i], r3det_results['all_boxes'][i]], axis=0)
        #print([len(box) for box in all_boxes_r])
        
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
        #all_boxes_r = []
        #for fr in nms_boxes:
        #    arr = fr.get_values()[:,[-1,0,1,2,3,4,5]]
        #    arr[:,0] += 1
        #    all_boxes_r.append(np.array(arr))
        #print([len(box) for box in all_boxes_r])
        
        # draw detections
        path_save = f'../../output/results/{self.cfgs.VERSION}_results/images'
        os.makedirs(path_save, exist_ok=True)
        
        for path,boxes in zip(imgs,all_boxes_r):
            img = cv2.imread(os.path.join(self.args.img_dir, path))
            
            for box in boxes:
                _,score,cx,cy,w,h,a = box
                if score<0.1:
                    continue
                box = cv2.boxPoints(((cx,cy),(w,h),a))
                box = np.int0(box)
                cv2.drawContours(img, [box], -1, (0,255,0), 2)
                cv2.putText(img, str(np.round(score,2)), (int(cx),int(cy)), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
            
            cv2.imwrite(os.path.join(path_save, path), img)
        
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.write_voc_results_file(all_boxes=all_boxes_r,
                                      test_imgid_list=real_test_imgname_list,
                                      det_save_dir=f'../../output/results/{self.cfgs.VERSION}_results')
        evaler.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':

    tester = TestUFRGSCELLGWD(cfgs)
    tester.eval()
