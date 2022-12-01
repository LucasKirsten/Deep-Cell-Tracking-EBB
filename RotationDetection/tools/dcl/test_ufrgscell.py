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

from libs.models.detectors.dcl import build_whole_network
from libs.val_libs.voc_eval_r import EVAL
from tools.test_ufrgscell_base import TestUFRGSCELL
from libs.configs import cfgs


class TestUFRGSCELLGWD(TestUFRGSCELL):

    def eval(self):
        dcl = build_whole_network.DetectionNetworkDCL(cfgs=self.cfgs,
                                                      is_training=False)

        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=dcl,
                                          image_ext=self.args.image_ext)

        imgs = os.listdir(self.args.img_dir)
        
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':

    tester = TestUFRGSCELLGWD(cfgs)
    tester.eval()
