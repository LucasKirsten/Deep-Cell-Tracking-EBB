import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
sys.path.append('../../..')

from utils.tools import makedirs
from libs.utils.coordinate_convert import backward_convert
from tqdm import tqdm

class_list = ['normal_cell']#, 'mitoses']

def clip_image(file_idx, image, width, height, stride_w, stride_h):
    min_pixel = 2

    shape = image.shape
    for start_h in range(0, shape[0], stride_h):
        for start_w in range(0, shape[1], stride_w):
            start_h_new = start_h
            start_w_new = start_w
            if start_h + height > shape[0]:
                start_h_new = shape[0] - height
            if start_w + width > shape[1]:
                start_w_new = shape[1] - width
            top_left_row = max(start_h_new, 0)
            top_left_col = max(start_w_new, 0)
            bottom_right_row = min(start_h + height, shape[0])
            bottom_right_col = min(start_w + width, shape[1])

            subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]


            if (subImage.shape[0] > 5 and subImage.shape[1] > 5):
                img = os.path.join(save_dir, "%s_%04d_%04d.png" % (file_idx, top_left_row, top_left_col))
                cv2.imwrite(img, subImage)


print('class_list', len(class_list))
# change here
path_root = sys.argv[-1]
raw_images_dir = path_root
save_dir = os.path.join(path_root, 'crop')

os.makedirs(save_dir, exist_ok=True)

images = [i for i in os.listdir(raw_images_dir) if 'jpg' in i]

print('found image', len(images))

min_length = 1e10
max_length = 1

img_h, img_w, stride_h, stride_w = 512, 512, 256, 256

pbar = tqdm(enumerate(images), total=len(images))
pbar.set_description('Croping images')
for idx, img in pbar:
    img_data = cv2.imread(os.path.join(raw_images_dir, img))

    clip_image(img.strip('.jpg'), img_data, img_w, img_h, stride_w, stride_h)
