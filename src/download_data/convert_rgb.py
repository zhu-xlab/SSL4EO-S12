import os
import rasterio
import cv2
import numpy as np
import random
from PIL import Image

ALL_BANDS_S2_L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
ALL_BANDS_S2_L1C = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
ALL_BANDS_S1_GRD = ['VV','VH']

S2C_MEAN = {
        'B1': 1612.9,
        'B2': 1397.6,
        'B3': 1322.3,
        'B4': 1373.1,
        'B5': 1561.0,
        'B6': 2108.4,
        'B7': 2390.7,
        'B8': 2318.7,
        'B8A': 2581.0,
        'B9': 837.7,
        'B10': 22.0,
        'B11': 2195.2,
        'B12': 1537.4}
S2C_STD = {
        'B1': 791.0,
        'B2': 854.3,
        'B3': 878.7,
        'B4': 1144.9,
        'B5': 1127.5,
        'B6': 1164.2,
        'B7': 1276.0,
        'B8': 1249.5,
        'B8A': 1345.9,
        'B9': 577.5,
        'B10': 47.5,
        'B11': 1340.0,
        'B12': 1142.9}

S2A_MEAN = {
        'B1': 756.4,
        'B2': 889.6,
        'B3': 1151.7,
        'B4': 1307.6,
        'B5': 1637.6,
        'B6': 2212.6,
        'B7': 2442.0,
        'B8': 2538.9,
        'B8A': 2602.9,
        'B9': 2666.8,
        'B11': 2388.8,
        'B12': 1821.5}
S2A_STD = {
        'B1': 1111.4,
        'B2': 1159.1,
        'B3': 1188.1,
        'B4': 1375.2,
        'B5': 1376.6,
        'B6': 1358.6,
        'B7': 1418.4,
        'B8': 1476.4,
        'B8A': 1439.9,
        'B9': 1582.1,
        'B11': 1460.7,
        'B12': 1352.2}

S1_MEAN = {'VV': -12.59, 'VH': -20.26}
S1_STD = {'VV': 5.26, 'VH': 5.91}


def normalize(img,mean,std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img



def get_array(patch_id, mode, RGB=False, norm=False):
    data_root_patch = os.path.join(root_dir, mode, patch_id)
    patch_seasons = os.listdir(data_root_patch)
    seasons = {}

    if mode=='s1':
        bands = ALL_BANDS_S1_GRD
        MEAN = S1_MEAN
        STD = S1_STD
    elif mode=='s2a':
        bands = ALL_BANDS_S2_L2A if RGB==False else RGB_BANDS
        MEAN = S2A_MEAN
        STD = S2A_STD       
    elif mode=='s2c':
        bands = ALL_BANDS_S2_L1C if RGB==False else RGB_BANDS
        MEAN = S2C_MEAN
        STD = S2C_STD
    
    for patch_id_season in patch_seasons:
        chs = []
        for i,band in enumerate(bands):
            patch_path = os.path.join(data_root_patch,patch_id_season,f'{band}.tif')
            with rasterio.open(patch_path) as dataset:
                ch = dataset.read(1)
                ch = cv2.resize(ch, dsize=(264, 264), interpolation=cv2.INTER_LINEAR_EXACT) # [264,264]
                
                if norm:
                    ch = normalize(ch,mean=MEAN[band],std=STD[band]) # uint8

                #coord = dataset.xy(0,0) # up left                    
            chs.append(ch)
        img = np.stack(chs, axis=-1) # [264,264,C]
        seasons[patch_id_season] = img

    return seasons



root_dir = './'
random.seed(42)
sample_ids = random.sample(range(0, 100), 10)

sample_inames = []
for id in sample_ids:
    iname = f'{id:07d}'
    sample_inames.append(iname)

print(sample_inames)

for index in sample_inames:
    #img_s1_4s = get_array(index, 's1')
    #img_s2a_4s = get_array(index, 's2a')
    img_s2c_4s = get_array(index, 's2c', RGB=True, norm=True)

    fdir = os.path.join('rgb_subset','s2c',index)
    if not os.path.isdir(fdir):
        os.makedirs(fdir,exist_ok=True)
        
    for t in img_s2c_4s.keys():
        img_t = img_s2c_4s[t]
        img_t = Image.fromarray(img_t,mode='RGB')

        img_t.save(os.path.join('rgb_subset','s2c',index,t+'.png'))

    #with h5py.File('ssl4eo-s12_h5/'+'s1/'+index+'.h5','w') as hf1:
    #    h5_s1 = hf1.create_dataset('array',data=img_s1_4s,shape=img_s1_4s.shape,chunks=True)
    #with h5py.File('ssl4eo-s12_h5/'+'s2a/'+index+'.h5','w') as hf2:
    #    h5_s2a = hf2.create_dataset('array',data=img_s2a_4s,shape=img_s2a_4s.shape,chunks=True)
    #with h5py.File('ssl4eo-s12_h5/'+'s2c/'+index+'.h5','w') as hf3:
    #    h5_s2c = hf3.create_dataset('array',data=img_s2c_4s,shape=img_s2c_4s.shape,chunks=True)    














