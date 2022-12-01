import os
from glob import glob
from tqdm import tqdm

path_pred = '/workdir/msc/RotationDetection/tools/gwd/test_dota/RetinaNet_DOTA_GWD'

path_save = os.path.join(path_pred, 'preds')
os.makedirs(path_save, exist_ok=True)
txt_preds = glob(os.path.join(path_pred, 'dota_res', '*.txt'))
for path in tqdm(txt_preds):
    label = os.path.split(path)[-1].split('.')[0].split('_')[1]
    
    with open(path, 'r') as file:
        preds = file.read().split('\n')
    for p in preds:
        values = p.split(' ')
        
        img_name = values[0]
        with open(os.path.join(path_save, img_name+'.txt'), 'a') as file:
            file.write(' '.join(values[2:]) + ' ' + label + '\n')