import os
import numpy as np
import pandas as pd
from numba import jit

# get OBB points from annotations
def parse_ann(ann):
    
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
def get_ann(path_annotations):
    
    annotations = {}
    for path in path_annotations:
        
        # open annotation path
        ann = open(path, 'r').read()
        ann = ann.split('\n')
        
        # get annotations
        boxes, labels = parse_ann(ann)
        
        annotations[path] = {'boxes':boxes, 'labels':labels}
        
    return annotations

def get_csv(path_annotations):
    
    annotations = _get_ann(path_annotations)
    print(annotations)
    
    csv_ann = {'path':[], 'x1':[], 'y1':[], 'x2':[], 'y2':[], 'x3':[], 'y3':[], 'x4':[], 'y4':[]}
    for img_name in annotations:
        for box in annotations[img_name]:
            csv_ann['path'].append(img_name)
            csv_ann['x1'].append(box[0][0])
            csv_ann['y1'].append(box[0][1])
            csv_ann['x2'].append(box[1][0])
            csv_ann['y2'].append(box[1][1])
            csv_ann['x3'].append(box[2][0])
            csv_ann['y3'].append(box[2][1])
            csv_ann['x4'].append(box[3][0])
            csv_ann['y4'].append(box[3][1])
            
    return pd.DataFrame(csv_ann)

#@jit(nopython=True, parallel=True)
def ann2gt(centers, reduce=16):
    displacement = []
    # placeholder for placing the centers
    ground_truth = np.zeros((centers.shape[0]//reduce,centers.shape[1]//reduce))

    # first map all unique center points and save the conflits
    conflits = []
    for gh,h in enumerate(range(reduce//2-1,centers.shape[0]-reduce//2,reduce)):
        for gw,w in enumerate(range(reduce//2-1,centers.shape[1]-reduce//2,reduce)):
            patch = centers[h-reduce//2:h+reduce//2, w-reduce//2:w+reduce//2]

            # get the number of center points
            nbr_centers = np.count_nonzero(patch)

            if nbr_centers==1:
                ground_truth[gh,gw] = 255
            
            if nbr_centers>1:
                conflits.append([[gh,gw], [h,w]])

    # iterate over patches with conflits
    for (gh,gw), (h,w) in conflits:
        # get the conflit points in the patch
        patch = centers[h-reduce//2:h+reduce//2, w-reduce//2:w+reduce//2]
        cnt_pts = np.where(patch>0)
        cnt_pts = [[cnt_pts[0][i],cnt_pts[1][i]] for i in range(len(cnt_pts))]
        
        # calculate the distance to the borders
        dleft, dright, dtop, dbottom = [],[],[],[]
        for pt in cnt_pts:
            dleft.append(pt[1])
            dright.append(ground_truth.shape[0]-pt[0])
            dtop.append(pt[0])
            dbottom.append(ground_truth.shape[1]-pt[1])
        
        # calculate the distance to the center and sort related to how close they are to the center
        cnt_dist = [np.square((p[0]-reduce/2.)**2. + (p[1]-reduce/2.)**2.) for p in cnt_pts]
        arg_pts = np.argsort(cnt_dist, kind='mergesort')
        
        # now fill the gaps related to the conflits
        for i in arg_pts:
            # verify the best displacement
            extreme = np.array([dleft[i], dright[i], dtop[i], dbottom[i]])
            extreme = np.argsort(extreme, kind='mergesort')
            found = False
            
            # try to fit in the sorted best direction
            found = False
            for k in range(ground_truth.shape[0]):
                for disp in extreme:
                    if disp==0: # left
                        if ground_truth[gh, gw-k]==0 and gw-k>0:
                            ground_truth[gh, gw-k] = 255
                            found = True
                    elif disp==1: # right
                        if gw+k<ground_truth.shape[1] and ground_truth[gh, gw+k]==0:
                            ground_truth[gh, gw+k] = 255
                            found = True
                    elif disp==2: # top
                        if ground_truth[gh-k, gw]==0 and gh-k>0:
                            ground_truth[gh-k, gw] = 255
                            found = True
                    else: # bottom
                        if gh+k<ground_truth.shape[0] and ground_truth[gh+k, gw]==0:
                            ground_truth[gh+k, gw] = 255
                            found = True
                    if found:
                        displacement.append(k)
                        break
                if found:
                    break
                    
    return ground_truth, displacement