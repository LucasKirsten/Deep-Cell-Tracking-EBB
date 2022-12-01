import os
import cv2
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import exposure

def norm(x):
    return np.uint8((x-np.min(x))/(np.max(x)-np.min(x))*255)

if __name__=='__main__':
    
    path_root = sys.argv[-3]
    extension = sys.argv[-2]
    augment   = float(sys.argv[-1])
    
    path_images = sorted(glob(path_root+'/*'+extension))
    
    mean,std = [],[]
    pbar = tqdm(enumerate(path_images), total=len(path_images))
    pbar.set_description('Converting images to jpg')
    for i,path in pbar:
        
        if path.endswith('.jpg'):
            continue

        img = cv2.imread(path, -1)
        
        img = norm(img)
        img = np.uint8(img)
        img = cv2.resize(img, (int(img.shape[1]*augment),int(img.shape[0]*augment)))
        
        ext = path.split('.')[-1]
        cv2.imwrite(path.replace(ext, 'jpg'), img)
        
        mean.append(np.mean(img, axis=(0,1)))
        std.append(np.std(img, axis=(0,1)))
        
    print(path_root)
    print('MEAN: ', np.mean(mean, axis=0))
    print('STD: ', np.mean(std, axis=0))