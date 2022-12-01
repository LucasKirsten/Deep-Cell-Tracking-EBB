# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
tf.enable_eager_execution()
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append("../")

import configs as cfgs
from libs.val_libs.voc_eval_r import EVAL

from test_ufrgscell_base import TestUFRGSCELL


class TestUFRGSCellDetector(TestUFRGSCELL):

    def eval(self):
        
        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, image_ext=self.args.image_ext)

        imgs = os.listdir(self.args.img_dir)
        real_test_imgname_list = [i.split(self.args.image_ext)[0] for i in imgs]

        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=self.args.test_annotation_path)


if __name__ == '__main__':
    
    tester = TestUFRGSCellDetector(cfgs)
    tester.eval()
