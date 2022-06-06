import numpy as np
import rasterio
import torch
import os
import cv2
import csv
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm



ALL_BANDS_S2_L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
ALL_BANDS_S2_L1C = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
ALL_BANDS_S1_GRD = ['VV','VH']


### band statistics: mean & std
# calculated from 50k data
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]

S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

# normalize: standardize + percentile
def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


### dataset class
class SSL4EO(torch.utils.data.Dataset):

    def __init__(self,root, normalize=False, mode=['s1','s2a','s2c'], dtype='uint8'):
        self.root = root
        self.normalize = normalize
        self.mode = mode
        self.dtype = dtype
        
        self.ids = os.listdir(os.path.join(self.root,self.mode[0]))
        self.length = len(self.ids)

    def __getitem__(self,index):
        
        if 's1' in self.mode:
            img_s1_4s = self.get_array(self.ids[index], 's1') # [4,2,264,264] float32 or uint8                
        else:
            img_s1_4s = None
            
        if 's2a' in self.mode:
            img_s2a_4s = self.get_array(self.ids[index], 's2a') # [4,12,264,264] int16 or uint8
        else:
            img_s2a_4s = None
            
        if 's2c' in self.mode:
            img_s2c_4s = self.get_array(self.ids[index], 's2c') # [4,13,264,264] int16 or uint8
        else:
            img_s2c_4s = None

        return img_s1_4s, img_s2a_4s, img_s2c_4s
        
    def __len__(self):    
        return self.length

    def get_array(self, patch_id, mode):
        data_root_patch = os.path.join(self.root, mode, patch_id)
        patch_seasons = os.listdir(data_root_patch)
        seasons = []

        if mode=='s1':
            bands = ALL_BANDS_S1_GRD
            MEAN = S1_MEAN
            STD = S1_STD
        elif mode=='s2a':
            bands = ALL_BANDS_S2_L2A
            MEAN = S2A_MEAN
            STD = S2A_STD            
        elif mode=='s2c':
            bands = ALL_BANDS_S2_L1C
            MEAN = S2C_MEAN
            STD = S2C_STD
            
        for patch_id_season in patch_seasons:
            chs = []
            for i,band in enumerate(bands):
                patch_path = os.path.join(data_root_patch,patch_id_season,f'{band}.tif')
                with rasterio.open(patch_path) as dataset:
                    ch = dataset.read(1)
                    ch = cv2.resize(ch, dsize=(264, 264), interpolation=cv2.INTER_LINEAR_EXACT) # [264,264]
                    #coord = dataset.xy(0,0) # up left
                    if self.normalize or (self.dtype=='uint8' and mode=='s1'):
                        ch = normalize(ch, MEAN[i], STD[i])
                        
                chs.append(ch)
            img = np.stack(chs, axis=0) # [C,264,264]
            seasons.append(img)
        img_4s = np.stack(seasons, axis=0) # [4,C,264,264]

        if self.normalize:
            return img_4s
        elif self.dtype=='uint8':
            if mode=='s1':
                return img_4s
            else:
                return (img_4s / 10000.0 * 255.0).astype('uint8')
        else:
            if mode=='s1':                    
                return img_4s.astype('float32')
            else:
                return img_4s.astype('int16')


            
class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def make_lmdb(dataset, lmdb_file, num_workers=6,mode=['s1','s2a','s2c']):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    #env = lmdb.open(lmdb_file, map_size=1099511627776,writemap=True) # continuously write to disk
    env = lmdb.open(lmdb_file, map_size=1099511627776)
    txn = env.begin(write=True)
    for index, (s1, s2a, s2c) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        if 's1' in mode:
            sample_s1 = np.array(s1)
        if 's2a' in mode:
            sample_s2a = np.array(s2a)
        if 's2c' in mode:
            sample_s2c = np.array(s2c)
            
        if mode==['s1','s2a','s2c']:
            obj = (sample_s1.tobytes(), sample_s1.shape, sample_s2a.tobytes(), sample_s2a.shape, sample_s2c.tobytes(), sample_s2c.shape)
        elif mode==['s1']:
            obj = (sample_s1.tobytes(), sample_s1.shape)
        elif mode==['s2a']:
            obj = (sample_s2a.tobytes(), sample_s2a.shape)
        elif mode==['s2c']:
            obj = (sample_s2c.tobytes(), sample_s2c.shape)
            
        txn.put(str(index).encode(), pickle.dumps(obj))            

        if index % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()
            
            
            
    
if __name__ == '__main__':

    import argparse
    import shutil
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--save_path',type=str)
    parser.add_argument('--make_lmdb_file',action='store_true',default=False)
    parser.add_argument('--frac',type=float,default=1.0)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--normalize',action='store_true',default=False)
    parser.add_argument('--mode', nargs='*', type=str, default=['s1','s2a','s2c'])
    parser.add_argument('--dtype',type=str, default='uint8')
    args = parser.parse_args()

    ### make lmdb dataset
    if args.make_lmdb_file:
        if os.path.isdir(args.save_path):
            shutil.rmtree(args.save_path)        
        train_dataset = SSL4EO(root=args.root, normalize=args.normalize, mode=args.mode, dtype=args.dtype)
        train_subset = random_subset(train_dataset,frac=args.frac,seed=42)

        make_lmdb(train_subset,args.save_path,num_workers=args.num_workers,mode=args.mode)

    ### check dataset class
    else:   
        train_dataset = SSL4EO(root=args.root, transform = None)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)
        i=0
        for idx, (s1,s2a,s2c) in tqdm(enumerate(train_loader),total=len(train_dataset)):
            if idx>0:
                break
            print(s1.shape, s1.dtype, s2a.shape,s2a.dtype, s2c.shape, s2c.dtype)

