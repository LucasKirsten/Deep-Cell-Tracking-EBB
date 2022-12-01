import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# adjust the OBBs to the image size
def _adjust_boxes(boxes, h, w):
    boxes[:,:,0] *= w
    boxes[:,:,1] *= h
    boxes = np.int0(boxes)
    return boxes

# get OBB points from annotations
def _parse_ann(ann):
    
    anns_boxes  = []
    anns_labels = []
    for box in ann:
        an = list(map(float, box.split(',')[1:-1]))
        lb = box.split(',')[-1]
        if len(an)>1:
            anns_boxes.append(np.array([
                [an[0],an[4]],
                [an[1],an[5]],
                [an[2],an[6]],
                [an[3],an[7]]]))
            lb = 'mitoses' if 'RoundCell' in lb else 'normal_cell'
            anns_labels.append(lb)
    return np.array(anns_boxes), anns_labels

# get all annotations
def _get_ann(path_annotations):
    
    annotations = {}
    for path in path_annotations:
        
        # open annotation path
        ann = open(path, 'r').read()
        ann = ann.split('\n')
        
        # get annotations
        boxes, labels = _parse_ann(ann)
        
        annotations[path] = {'boxes':boxes, 'labels':labels}
        
    return annotations

# convert annotations to dota format
def convert2dota(path_imgs, path_ann, path_save):
    
    anns = _get_ann(path_ann)
    for i in tqdm(range(len(path_imgs))):
        path_img = path_imgs[i]
        ann = anns[path_ann[i]]
        
        img = cv2.imread(path_img)
        h,w,_ = img.shape
        
        boxes = _adjust_boxes(np.copy(ann['boxes']), h, w)
        labels = ann['labels']
        
        label_name = os.path.split(path_ann[i])[-1]
        
        with open(os.path.join(path_save, label_name), 'w') as file:
            file.write('imagesource:UFRGS\n')
            file.write('gsd:0\n')
            for box,lb in zip(boxes, labels):
                box = map(str, box.reshape(-1))
                box = ' '.join(list(box))
                lb = 'mitoses' if 'RoundCell' in lb else 'normal_cell'
                
                file.write(box + f' {lb} 0\n')
    
if __name__=='__main__':
    
    # change here
    path_root = '/workdir/datasets/msc/UFRGS_CELL_2classes/test'
    path_img  = os.path.join(path_root, 'imgs')
    path_ann  = os.path.join(path_root, 'annotations', 'alpr')
    path_save = os.path.join(path_root, 'annotations', 'dota_format')
    
    os.makedirs(path_save, exist_ok=True)
    path_imgs = sorted(glob(os.path.join(path_img, '*')))
    path_ann  = sorted(glob(os.path.join(path_ann, '*')))
    convert2dota(path_imgs, path_ann, path_save)