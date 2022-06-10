from pathlib import Path
from itertools import product

from torch.utils.data import Dataset
import rasterio
import numpy as np
# from PIL import Image

# import random
# import cv2
# import torch


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 
             'B05', 'B06', 'B07', 'B08', 
             'B8A', 'B09', 'B10', 'B11', 
             'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

QUANTILES_RGB = {
    'min_q': {
        'B02': 885.0,
        'B03': 667.0,
        'B04': 426.0
    },
    'max_q': {
        'B02': 2620.0,
        'B03': 2969.0,
        'B04': 3698.0
    }
}

QUANTILES_ALL = {
    'min_q': {
        'B01': 1194.0, 'B02': 885.0,  'B03': 667.0,  'B04': 426.0,
        'B05': 392.0,  'B06': 358.0,  'B07': 349.0,  'B08': 290.0,
        'B8A': 310.0,  'B09': 96.0,   'B10': 7.0,    'B11': 155.0,
        'B12': 109.0
    },
    'max_q': {
        'B01': 2456.0, 'B02': 2620.0, 'B03': 2969.0, 'B04': 3698.0,
        'B05': 3803.0, 'B06': 3994.0, 'B07': 4261.0, 'B08': 4141.0,
        'B8A': 4435.0, 'B09': 1589.0, 'B10': 51.0,   'B11': 5043.0,
        'B12': 4238.0
    }
}

def read_image(path, bands, normalize=True, value_discard=True):
    # # original: read from 'imgs_1' and 'imgs_2' dirs
    # patch_id = next(path.iterdir()).name[:-8]
    # # get img shape for interpolation
    # img_shp = rasterio.open(path / f'{patch_id}_B02.tif').read(1).shape
    channels = []
    QUANTILES = QUANTILES_RGB if len(bands)==3 else QUANTILES_ALL
    for b in bands:
        # # original: read from 'imgs_1' and 'imgs_2' dirs
        # ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
        ch = rasterio.open(path / f'{b}.tif').read(1)
        
        # # interpolation
        # ch = cv2.resize(ch,img_shp,interpolation=cv2.INTER_CUBIC)
        
        if normalize:
            if value_discard:
                min_v = QUANTILES['min_q'][b]
                max_v = QUANTILES['max_q'][b]
                ch = (ch - min_v) / (max_v - min_v)
            else:
                ch = ch/10000
            ch = np.clip(ch, 0, 1)
            ch = (ch * 255).astype(np.uint8)
        channels.append(ch)
    img = np.dstack(channels)
    # # original: convert np.array to Image object
    # img = Image.fromarray(img)
    return img


    

class ChangeDetectionDataset(Dataset):

    def __init__(self, root, split='all', bands=None, transform=None, 
                 value_discard=True, patch_size=96):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else ALL_BANDS
        self.transform = transform
        self.value_discard = value_discard

        with open(self.root / f'{split}.txt') as f:
            names = f.read().strip().split(',')

        self.samples = []
        for name in names:
            fp = next((self.root / name / 'imgs_1_rect').glob(f'*{self.bands[0]}*'))
            img = rasterio.open(fp)
            limits = product(range(0, img.width, patch_size), range(0, img.height, patch_size))
            for l in limits:
                if l[0] + patch_size < img.width: 
                    if l[1] + patch_size < img.height:
                            self.samples.append((self.root / name, (l[0], l[1], l[0] + patch_size, l[1] + patch_size)))

    def __getitem__(self, index):
        path, limits = self.samples[index]     

        img_1 = read_image(path / 'imgs_1_rect', self.bands, value_discard=self.value_discard)    # Image -> np.array, type: unit8
        img_2 = read_image(path / 'imgs_2_rect', self.bands, value_discard=self.value_discard)    # Image -> np.array, type: unit8
        # # original: read cm as an Image object
        # cm = Image.open(path / 'cm' / 'cm.png').convert('L')
        cm = rasterio.open(path / 'cm' / 'cm.png').read(1).astype(np.uint8)     # np.array, type: unit8

        # # using crop from PIL for 3 bands
        # img_1 = img_1.crop(limits)
        # img_2 = img_2.crop(limits)
        # cm = cm.crop(limits)
        
        # crop for 13 bands
        img_1 = img_1[limits[1]:limits[3],limits[0]:limits[2],:]
        img_2 = img_2[limits[1]:limits[3],limits[0]:limits[2],:]
        cm = cm[limits[1]:limits[3],limits[0]:limits[2]]

        if self.transform is not None:
            img_1, img_2, cm = self.transform(img_1, img_2, cm)
        
        return img_1, img_2, cm

    def __len__(self):
        return len(self.samples)
