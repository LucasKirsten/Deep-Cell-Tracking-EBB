# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
from multiprocessing import Queue, Process
sys.path.append("../")

from utils import tools
from libs.label_name_dict.label_dict import LabelMap
from libs.utils.draw_box_in_img import DrawBox
from libs.utils.coordinate_convert import forward_convert, backward_convert
from libs.utils import nms_rotate
from libs.utils import nms
from libs.utils.rotate_polygon_nms import rotate_gpu_nms
from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


def parse_args():
    parser = argparse.ArgumentParser('Start testing.')

    parser.add_argument('--test_dir', dest='test_dir',
                        help='evaluate imgs dir ',
                        default='/data/DOTA/test/images/', type=str)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu id',
                        default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--eval_num', dest='eval_num',
                        help='the num of eval imgs',
                        default=np.inf, type=int)
    parser.add_argument('--show_box', '-s', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')
    parser.add_argument('--flip_img', '-f', default=False,
                        action='store_true')
    parser.add_argument('--num_imgs', dest='num_imgs',
                        help='test image number',
                        default=np.inf, type=int)
    parser.add_argument('--h_len', dest='h_len',
                        help='image height',
                        default=600, type=int)
    parser.add_argument('--w_len', dest='w_len',
                        help='image width',
                        default=600, type=int)
    parser.add_argument('--h_overlap', dest='h_overlap',
                        help='height overlap',
                        default=150, type=int)
    parser.add_argument('--w_overlap', dest='w_overlap',
                        help='width overlap',
                        default=150, type=int)
    args = parser.parse_args()
    return args


class TestDOTA(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        label_map = LabelMap(cfgs)
        self.name_label_map, self.label_name_map = label_map.name2label(), label_map.label2name()

    def worker(self, gpu_id, images, det_net, result_queue):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
        img_batch = tf.cast(img_plac, tf.float32)

        pretrain_zoo = PretrainModelZoo()
        if self.cfgs.NET_NAME in pretrain_zoo.pth_zoo or self.cfgs.NET_NAME in pretrain_zoo.mxnet_zoo:
            img_batch = (img_batch / 255 - tf.constant(self.cfgs.PIXEL_MEAN_)) / tf.constant(self.cfgs.PIXEL_STD)
        else:
            img_batch = img_batch - tf.constant(self.cfgs.PIXEL_MEAN)

        img_batch = tf.expand_dims(img_batch, axis=0)

        detection_boxes_h, detection_scores_h, detection_category_h, \
        detection_boxes_r, detection_scores_r, detection_category_r = det_net.build_whole_detection_network(
            input_img_batch=img_batch)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = det_net.get_restorer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model %d ...' % gpu_id)

            for img_path in images:

                # if 'P0016' not in img_path:
                #     continue

                img = cv2.imread(img_path)

                box_res, label_res, score_res = [], [], []
                box_res_rotate, label_res_rotate, score_res_rotate = [], [], []

                imgH = img.shape[0]
                imgW = img.shape[1]

                img_short_side_len_list = self.cfgs.IMG_SHORT_SIDE_LEN if isinstance(self.cfgs.IMG_SHORT_SIDE_LEN, list) else [
                    self.cfgs.IMG_SHORT_SIDE_LEN]
                img_short_side_len_list = [img_short_side_len_list[0]] if not self.args.multi_scale else img_short_side_len_list

                if imgH < self.args.h_len:
                    temp = np.zeros([self.args.h_len, imgW, 3], np.float32)
                    temp[0:imgH, :, :] = img
                    img = temp
                    imgH = self.args.h_len

                if imgW < self.args.w_len:
                    temp = np.zeros([imgH, self.args.w_len, 3], np.float32)
                    temp[:, 0:imgW, :] = img
                    img = temp
                    imgW = self.args.w_len

                for hh in range(0, imgH, self.args.h_len - self.args.h_overlap):
                    if imgH - hh - 1 < self.args.h_len:
                        hh_ = imgH - self.args.h_len
                    else:
                        hh_ = hh
                    for ww in range(0, imgW, self.args.w_len - self.args.w_overlap):
                        if imgW - ww - 1 < self.args.w_len:
                            ww_ = imgW - self.args.w_len
                        else:
                            ww_ = ww
                        src_img = img[hh_:(hh_ + self.args.h_len), ww_:(ww_ + self.args.w_len), :]

                        for short_size in img_short_side_len_list:
                            max_len = self.cfgs.IMG_MAX_LENGTH
                            if self.args.h_len < self.args.w_len:
                                new_h, new_w = short_size, min(int(short_size * float(self.args.w_len) / self.args.h_len), max_len)
                            else:
                                new_h, new_w = min(int(short_size * float(self.args.h_len) / self.args.w_len), max_len), short_size
                            img_resize = cv2.resize(src_img, (new_w, new_h))

                            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
                            det_boxes_r_, det_scores_r_, det_category_r_ = \
                                sess.run(
                                    [img_batch, detection_boxes_h, detection_scores_h, detection_category_h,
                                     detection_boxes_r, detection_scores_r, detection_category_r],
                                    feed_dict={img_plac: img_resize[:, :, ::-1]}
                                )

                            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                            src_h, src_w = src_img.shape[0], src_img.shape[1]

                            if len(det_boxes_h_) > 0:
                                det_boxes_h_[:, 0::2] *= (src_w / resized_w)
                                det_boxes_h_[:, 1::2] *= (src_h / resized_h)
                                for ii in range(len(det_boxes_h_)):
                                    box = det_boxes_h_[ii]
                                    box[0::2] = box[0::2] + ww_
                                    box[1::2] = box[1::2] + hh_
                                    box_res.append(box)
                                    label_res.append(det_category_h_[ii])
                                    score_res.append(det_scores_h_[ii])

                            if len(det_boxes_r_) > 0:
                                det_boxes_r_ = forward_convert(det_boxes_r_, False)
                                det_boxes_r_[:, 0::2] *= (src_w / resized_w)
                                det_boxes_r_[:, 1::2] *= (src_h / resized_h)

                                for ii in range(len(det_boxes_r_)):
                                    box_rotate = det_boxes_r_[ii]
                                    box_rotate[0::2] = box_rotate[0::2] + ww_
                                    box_rotate[1::2] = box_rotate[1::2] + hh_
                                    box_res_rotate.append(box_rotate)
                                    label_res_rotate.append(det_category_r_[ii])
                                    score_res_rotate.append(det_scores_r_[ii])

                            if self.args.flip_img:
                                det_boxes_h_flip, det_scores_h_flip, det_category_h_flip, \
                                det_boxes_r_flip, det_scores_r_flip, det_category_r_flip = \
                                    sess.run(
                                        [detection_boxes_h, detection_scores_h, detection_category_h,
                                         detection_boxes_r, detection_scores_r, detection_category_r],
                                        feed_dict={img_plac: cv2.flip(img_resize, flipCode=1)[:, :, ::-1]}
                                    )

                                if len(det_boxes_h_) > 0:
                                    det_boxes_h_flip[:, 0::2] *= (src_w / resized_w)
                                    det_boxes_h_flip[:, 1::2] *= (src_h / resized_h)
                                    for ii in range(len(det_boxes_h_flip)):
                                        box = det_boxes_h_flip[ii]
                                        box[0::2] = src_w - box[0::2] + ww_
                                        box[1::2] = box[1::2] + hh_
                                        box_res.append(box)
                                        label_res.append(det_category_h_flip[ii])
                                        score_res.append(det_scores_h_flip[ii])

                                if len(det_boxes_r_flip) > 0:
                                    det_boxes_r_flip = forward_convert(det_boxes_r_flip, False)
                                    det_boxes_r_flip[:, 0::2] *= (src_w / resized_w)
                                    det_boxes_r_flip[:, 1::2] *= (src_h / resized_h)

                                    for ii in range(len(det_boxes_r_flip)):
                                        box_rotate = det_boxes_r_flip[ii]
                                        box_rotate[0::2] = (src_w - box_rotate[0::2]) + ww_
                                        box_rotate[1::2] = box_rotate[1::2] + hh_
                                        box_res_rotate.append(box_rotate)
                                        label_res_rotate.append(det_category_r_flip[ii])
                                        score_res_rotate.append(det_scores_r_flip[ii])

                                det_boxes_h_flip, det_scores_h_flip, det_category_h_flip,\
                                det_boxes_r_flip, det_scores_r_flip, det_category_r_flip = \
                                    sess.run(
                                        [detection_boxes_h, detection_scores_h, detection_category_h,
                                         detection_boxes_r, detection_scores_r, detection_category_r],
                                        feed_dict={img_plac: cv2.flip(img_resize, flipCode=0)[:, :, ::-1]}
                                    )

                                if len(det_boxes_h_) > 0:
                                    det_boxes_h_flip[:, 0::2] *= (src_w / resized_w)
                                    det_boxes_h_flip[:, 1::2] *= (src_h / resized_h)
                                    for ii in range(len(det_boxes_h_flip)):
                                        box = det_boxes_h_flip[ii]
                                        box[0::2] = box[0::2] + ww_
                                        box[1::2] = src_h - box[1::2] + hh_
                                        box_res.append(box)
                                        label_res.append(det_category_h_flip[ii])
                                        score_res.append(det_scores_h_flip[ii])

                                if len(det_boxes_r_flip) > 0:
                                    det_boxes_r_flip = forward_convert(det_boxes_r_flip, False)
                                    det_boxes_r_flip[:, 0::2] *= (src_w / resized_w)
                                    det_boxes_r_flip[:, 1::2] *= (src_h / resized_h)

                                    for ii in range(len(det_boxes_r_flip)):
                                        box_rotate = det_boxes_r_flip[ii]
                                        box_rotate[0::2] = box_rotate[0::2] + ww_
                                        box_rotate[1::2] = (src_h - box_rotate[1::2]) + hh_
                                        box_res_rotate.append(box_rotate)
                                        label_res_rotate.append(det_category_r_flip[ii])
                                        score_res_rotate.append(det_scores_r_flip[ii])

                box_res = np.array(box_res)
                label_res = np.array(label_res)
                score_res = np.array(score_res)
                box_res_rotate = np.array(box_res_rotate)
                label_res_rotate = np.array(label_res_rotate)
                score_res_rotate = np.array(score_res_rotate)

                box_res_rotate_, label_res_rotate_, score_res_rotate_ = [], [], []
                box_res_, label_res_, score_res_ = [], [], []

                threshold_r = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
                               'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.2, 'plane': 0.3,
                               'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
                               'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3,
                               'container-crane': 0.05, 'airport': 0.1, 'helipad': 0.1}

                threshold_h = {'roundabout': 0.35, 'tennis-court': 0.35, 'swimming-pool': 0.4, 'storage-tank': 0.3,
                               'soccer-ball-field': 0.3, 'small-vehicle': 0.4, 'ship': 0.35, 'plane': 0.35,
                               'large-vehicle': 0.4, 'helicopter': 0.4, 'harbor': 0.3, 'ground-track-field': 0.4,
                               'bridge': 0.3, 'basketball-court': 0.4, 'baseball-diamond': 0.3,
                               'container-crane': 0.3, 'airport': 0.2, 'helipad': 0.2}

                for sub_class in range(1, self.cfgs.CLASS_NUM + 1):
                    index = np.where(label_res_rotate == sub_class)[0]
                    if len(index) == 0:
                        continue
                    tmp_boxes_r = box_res_rotate[index]
                    tmp_label_r = label_res_rotate[index]
                    tmp_score_r = score_res_rotate[index]

                    tmp_boxes_r_ = backward_convert(tmp_boxes_r, False)

                    # try:
                    #     inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r_),
                    #                                     scores=np.array(tmp_score_r),
                    #                                     iou_threshold=threshold[self.label_name_map[sub_class]],
                    #                                     max_output_size=5000)
                    #
                    # except:
                    tmp_boxes_r_ = np.array(tmp_boxes_r_)
                    tmp = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                    tmp[:, 0:-1] = tmp_boxes_r_
                    tmp[:, -1] = np.array(tmp_score_r)
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r_.shape[0], ) / 1000
                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                         float(threshold_r[self.label_name_map[sub_class]]), 0)

                    box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                    score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                    label_res_rotate_.extend(np.array(tmp_label_r)[inx])

                for sub_class in range(1, self.cfgs.CLASS_NUM + 1):
                    index = np.where(label_res == sub_class)[0]
                    if len(index) == 0:
                        continue
                    tmp_boxes_h = box_res[index]
                    tmp_label_h = label_res[index]
                    tmp_score_h = score_res[index]

                    tmp_boxes_h = np.array(tmp_boxes_h)
                    tmp = np.zeros([tmp_boxes_h.shape[0], tmp_boxes_h.shape[1] + 1])
                    tmp[:, 0:-1] = tmp_boxes_h
                    tmp[:, -1] = np.array(tmp_score_h)

                    inx = nms.py_cpu_nms(dets=np.array(tmp, np.float32),
                                         thresh=float(threshold_h[self.label_name_map[sub_class]]),
                                         max_output_size=5000)

                    box_res_.extend(np.array(tmp_boxes_h)[inx])
                    score_res_.extend(np.array(tmp_score_h)[inx])
                    label_res_.extend(np.array(tmp_label_h)[inx])

                result_dict = {'boxes_h': np.array(box_res_), 'scores_h': np.array(score_res_),
                               'labels_h': np.array(label_res_), 'boxes_r': np.array(box_res_rotate_),
                               'scores_r': np.array(score_res_rotate_),
                               'labels_r': np.array(label_res_rotate_), 'image_id': img_path}
                result_queue.put_nowait(result_dict)

    def test_dota(self, det_net, real_test_img_list, txt_name):

        save_path = os.path.join('./test_dota', self.cfgs.VERSION)

        nr_records = len(real_test_img_list)
        pbar = tqdm(total=nr_records)
        gpu_num = len(self.args.gpus.strip().split(','))

        nr_image = math.ceil(nr_records / gpu_num)
        result_queue = Queue(500)
        procs = []

        for i, gpu_id in enumerate(self.args.gpus.strip().split(',')):
            start = i * nr_image
            end = min(start + nr_image, nr_records)
            split_records = real_test_img_list[start:end]
            proc = Process(target=self.worker, args=(int(gpu_id), split_records, det_net, result_queue))
            print('process:%d, start:%d, end:%d' % (i, start, end))
            proc.start()
            procs.append(proc)

        for i in range(nr_records):
            res = result_queue.get()

            if self.args.show_box:

                nake_name = res['image_id'].split('/')[-1]
                tools.makedirs(os.path.join(save_path, 'dota_img_vis_r'))
                draw_path_r = os.path.join(save_path, 'dota_img_vis_r', nake_name)

                draw_img = np.array(cv2.imread(res['image_id']), np.float32)
                detected_boxes = backward_convert(res['boxes_r'], with_label=False)

                detected_indices = res['scores_r'] >= self.cfgs.VIS_SCORE
                detected_scores = res['scores_r'][detected_indices]
                detected_boxes = detected_boxes[detected_indices]
                detected_categories = res['labels_r'][detected_indices]

                drawer = DrawBox(self.cfgs)

                final_detections = drawer.draw_boxes_with_label_and_scores(draw_img,
                                                                           boxes=detected_boxes,
                                                                           labels=detected_categories,
                                                                           scores=detected_scores,
                                                                           method=1,
                                                                           is_csl=True,
                                                                           in_graph=False)
                cv2.imwrite(draw_path_r, final_detections)

                tools.makedirs(os.path.join(save_path, 'dota_img_vis_h'))
                draw_path_h = os.path.join(save_path, 'dota_img_vis_h', nake_name)
                detected_indices = res['scores_h'] >= self.cfgs.VIS_SCORE
                detected_scores = res['scores_h'][detected_indices]
                detected_boxes = res['boxes_h'][detected_indices]
                detected_categories = res['labels_h'][detected_indices]

                final_detections = drawer.draw_boxes_with_label_and_scores(draw_img,
                                                                           boxes=detected_boxes,
                                                                           labels=detected_categories,
                                                                           scores=detected_scores,
                                                                           method=0,
                                                                           in_graph=False)
                cv2.imwrite(draw_path_h, final_detections)

            else:
                CLASS_DOTA = self.name_label_map.keys()
                write_handle_r, write_handle_h = {}, {}

                tools.makedirs(os.path.join(save_path, 'dota_res_r'))
                tools.makedirs(os.path.join(save_path, 'dota_res_h'))
                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_r[sub_class] = open(os.path.join(save_path, 'dota_res_r', 'Task1_%s.txt' % sub_class), 'a+')
                    write_handle_h[sub_class] = open(os.path.join(save_path, 'dota_res_h', 'Task2_%s.txt' % sub_class), 'a+')

                for i, rbox in enumerate(res['boxes_r']):
                    command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (res['image_id'].split('/')[-1].split('.')[0],
                                                                                     res['scores_r'][i],
                                                                                     rbox[0], rbox[1], rbox[2], rbox[3],
                                                                                     rbox[4], rbox[5], rbox[6], rbox[7],)
                    write_handle_r[self.label_name_map[res['labels_r'][i]]].write(command)
                for i, hbox in enumerate(res['boxes_h']):
                    command = '%s %.3f %.1f %.1f %.1f %.1f\n' % (res['image_id'].split('/')[-1].split('.')[0],
                                                                 res['scores_h'][i],
                                                                 hbox[0], hbox[1], hbox[2], hbox[3])
                    write_handle_h[self.label_name_map[res['labels_h'][i]]].write(command)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_r[sub_class].close()
                    write_handle_h[sub_class].close()

                fw = open(txt_name, 'a+')
                fw.write('{}\n'.format(res['image_id'].split('/')[-1]))
                fw.close()

            pbar.set_description("Test image %s" % res['image_id'].split('/')[-1])

            pbar.update(1)

        for p in procs:
            p.join()

    def get_test_image(self):
        txt_name = '{}.txt'.format(self.cfgs.VERSION)
        if not self.args.show_box:
            if not os.path.exists(txt_name):
                fw = open(txt_name, 'w')
                fw.close()

            fr = open(txt_name, 'r')
            img_filter = fr.readlines()
            print('****************************' * 3)
            print('Already tested imgs:', img_filter)
            print('****************************' * 3)
            fr.close()

            test_imgname_list = [os.path.join(self.args.test_dir, img_name) for img_name in os.listdir(self.args.test_dir)
                                 if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')) and
                                 (img_name + '\n' not in img_filter)]
        else:
            test_imgname_list = [os.path.join(self.args.test_dir, img_name) for img_name in os.listdir(self.args.test_dir)
                                 if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]

        assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                            ' Note that, we only support img format of (.jpg, .png, and .tiff) '

        if self.args.num_imgs == np.inf:
            real_test_img_list = test_imgname_list
        else:
            real_test_img_list = test_imgname_list[: self.args.num_imgs]

        return real_test_img_list


