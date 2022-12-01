# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
sys.path.append("../../")

from libs.models.detectors.retinanet import build_whole_network
from tools.test_mlt_base import TestMLT
from libs.configs import cfgs


class TestMLTRetinaNet(TestMLT):

    def eval(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        real_test_img_list = self.get_test_image()

        retinanet = build_whole_network.DetectionNetworkRetinaNet(cfgs=self.cfgs,
                                                                  is_training=False)
        self.test_mlt(det_net=retinanet, real_test_img_list=real_test_img_list, txt_name=txt_name)

        if not self.args.show_box:
            os.remove(txt_name)

if __name__ == '__main__':

    tester = TestMLTRetinaNet(cfgs)
    tester.eval()


