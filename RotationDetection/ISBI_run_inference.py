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

from importlib import import_module
dataset = sys.argv[-1]

cfgs = import_module(f'libs.configs.ISBI.{dataset}_all')
DetectionNetwork = import_module(f'libs.models.detectors.r2cnn.build_whole_network').DetectionNetworkR2CNN

from libs.val_libs.voc_eval_r import EVAL
from tools.test_ufrgscell_base import TestUFRGSCELL

class _TestUFRGSCELL(TestUFRGSCELL):

    def eval(self):
        net = DetectionNetwork(cfgs=self.cfgs, is_training=False)
        
        imgs = os.listdir(self.args.img_dir)
        image_ext = self.args.image_ext
        all_boxes_r = self.eval_with_plac(img_dir=self.args.img_dir, det_net=net,
                                          image_ext=image_ext)
        
        real_test_imgname_list = [i.split(image_ext)[0] for i in imgs if i.endswith(image_ext)]
        
        print(10 * "**")
        print('rotation eval:')
        evaler = EVAL(self.cfgs)
        evaler.write_voc_results_file(all_boxes=all_boxes_r,
                                      test_imgid_list=real_test_imgname_list,
                                      det_save_dir=f'./{dataset}_results')


if __name__ == '__main__':

    tester = _TestUFRGSCELL(cfgs)
    tester.eval()
