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
import math
from tqdm import tqdm
from tensorflow.python.tools import freeze_graph

sys.path.append("../../")

from libs.models.detectors.r3det import build_whole_network_pb
from libs.configs import cfgs
from libs.utils.draw_box_in_img import DrawBox
from libs.utils.nms_rotate import nms_rotate_cpu
from libs.models.anchor_heads.generate_anchors import GenerateAnchors
from utils import tools

CKPT_PATH = '../../output/trained_weights/{}/HRSC2016_199999model.ckpt'.format(cfgs.VERSION)
OUT_DIR = '../../output/Pbs'
PB_NAME = 'R3Det.pb'


class ExportPb(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs

    def build_detection_graph(self):
        r3det = build_whole_network_pb.DetectionNetworkR3Det(cfgs=self.cfgs, is_training=False)

        img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3], name='input_img')  # is RGB. not BGR
        img_batch = tf.cast(img_plac, tf.float32)

        if self.cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d',
                                  'resnet152_v1b', 'resnet101_v1b', 'resnet50_v1b', 'resnet34_v1b', 'resnet18_v1b']:
            img_batch = (img_batch / 255 - tf.constant(self.cfgs.PIXEL_MEAN_)) / tf.constant(self.cfgs.PIXEL_STD)
        else:
            img_batch = img_batch - tf.constant(self.cfgs.PIXEL_MEAN)

        img_batch = tf.expand_dims(img_batch, axis=0)

        box_pred, cls_prob, proposal = r3det.build_whole_detection_network(input_img_batch=img_batch)

        dets = tf.concat([tf.reshape(box_pred, [-1, 5]),
                          tf.reshape(cls_prob, [-1, self.cfgs.CLASS_NUM]),
                          tf.reshape(proposal, [-1, 5])], axis=1, name='DetResults')

        return dets

    def export_frozenPB(self):

        tf.reset_default_graph()

        dets = self.build_detection_graph()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("we have restred the weights from =====>>\n", CKPT_PATH)
            saver.restore(sess, CKPT_PATH)

            tf.train.write_graph(sess.graph_def, OUT_DIR, PB_NAME)
            freeze_graph.freeze_graph(input_graph=os.path.join(OUT_DIR, PB_NAME),
                                      input_saver='',
                                      input_binary=False,
                                      input_checkpoint=CKPT_PATH,
                                      output_node_names="DetResults",
                                      restore_op_name="save/restore_all",
                                      filename_tensor_name='save/Const:0',
                                      output_graph=os.path.join(OUT_DIR, PB_NAME.replace('.pb', '_Frozen.pb')),
                                      clear_devices=False,
                                      initializer_nodes='')

    def load_graph(self, frozen_graph_file):

        # we parse the graph_def file
        with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # we load the graph_def in the default graph

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                name="",
                                op_dict=None,
                                producer_op_list=None)
        return graph

    def rbbox_transform_inv(self, boxes, deltas, scale_factors=None):
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        dtheta = deltas[:, 4]

        if scale_factors:
            dx /= scale_factors[0]
            dy /= scale_factors[1]
            dw /= scale_factors[2]
            dh /= scale_factors[3]
            dtheta /= scale_factors[4]

        pred_ctr_x = dx * boxes[:, 2] + boxes[:, 0]
        pred_ctr_y = dy * boxes[:, 3] + boxes[:, 1]
        pred_w = np.exp(dw) * boxes[:, 2]
        pred_h = np.exp(dh) * boxes[:, 3]

        pred_theta = dtheta * 180 / np.pi + boxes[:, 4]

        return np.transpose(np.stack([pred_ctr_x, pred_ctr_y,
                                      pred_w, pred_h, pred_theta]))

    def postprocess_detctions(self, rpn_bbox_pred, rpn_cls_prob, anchors):

        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            scores = rpn_cls_prob[:, j]
            indices = scores > self.cfgs.VIS_SCORE

            anchors_ = anchors[indices]
            rpn_bbox_pred_ = rpn_bbox_pred[indices]
            scores = scores[indices]

            boxes_pred = self.rbbox_transform_inv(boxes=anchors_, deltas=rpn_bbox_pred_)

            nms_indices = nms_rotate_cpu(boxes=np.array(boxes_pred),
                                         scores=np.array(scores),
                                         iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                         max_output_size=20)

            tmp_boxes_pred = boxes_pred[nms_indices].reshape([-1, 5])
            tmp_scores = scores[nms_indices].reshape([-1, ])

            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(np.ones_like(scores) * (j + 1))

        return_boxes_pred = np.concatenate(return_boxes_pred, axis=0)
        return_scores = np.concatenate(return_scores, axis=0)
        return_labels = np.concatenate(return_labels, axis=0)

        return return_boxes_pred, return_scores, return_labels

    def test_pb(self, frozen_graph_path, test_dir):

        graph = self.load_graph(frozen_graph_path)
        print("we are testing ====>>>>", frozen_graph_path)

        img = graph.get_tensor_by_name("input_img:0")
        dets = graph.get_tensor_by_name("DetResults:0")

        with tf.Session(graph=graph) as sess:
            for img_path in os.listdir(test_dir):
                print(img_path)
                a_img = cv2.imread(os.path.join(test_dir, img_path))[:, :, ::-1]

                raw_h, raw_w = a_img.shape[0], a_img.shape[1]

                short_size, max_len = self.cfgs.IMG_SHORT_SIDE_LEN, cfgs.IMG_MAX_LENGTH
                if raw_h < raw_w:
                    new_h, new_w = short_size, min(int(short_size * float(raw_w) / raw_h), max_len)
                else:
                    new_h, new_w = min(int(short_size * float(raw_h) / raw_w), max_len), short_size
                img_resize = cv2.resize(a_img, (new_w, new_h))
                dets_val = sess.run(dets, feed_dict={img: img_resize[:, :, ::-1]})

                bbox_pred, cls_prob, proposal = dets_val[:, :5], dets_val[:, 5:(5+self.cfgs.CLASS_NUM)], \
                                                dets_val[:, (5+self.cfgs.CLASS_NUM):]

                detected_boxes, detected_scores, detected_categories = self.postprocess_detctions(bbox_pred, cls_prob, proposal)

                if True:
                    # detected_indices = det_scores_r_ >= self.cfgs.VIS_SCORE
                    # detected_scores = det_scores_r_[detected_indices]
                    # detected_boxes = det_boxes_r_[detected_indices]
                    # detected_categories = det_category_r_[detected_indices]

                    drawer = DrawBox(self.cfgs)

                    det_detections_r = drawer.draw_boxes_with_label_and_scores(img_resize[:, :, ::-1],
                                                                               boxes=detected_boxes,
                                                                               labels=detected_categories,
                                                                               scores=detected_scores,
                                                                               method=1,
                                                                               in_graph=True)

                    save_dir = os.path.join('test_pb', self.cfgs.VERSION, 'pb_img_vis')
                    tools.makedirs(save_dir)

                    cv2.imwrite(save_dir + '/{}'.format(img_path),
                                det_detections_r[:, :, ::-1])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    exporter = ExportPb(cfgs)
    exporter.export_frozenPB()
    exporter.test_pb('../../output/Pbs/R3Det_Frozen.pb',
                     '/data/dataset/HRSC2016/HRSC2016/Test/AllImages')
