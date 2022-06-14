import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm
import pdb

### band statistics: mean & std
# calculated from 50k subset
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]

S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
    
    

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


class LMDBDataset(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, s1_transform=None, s2a_transform=None, s2c_transform=None, subset=None, normalize=False, mode=['s1','s2a','s2c'], dtype='raw'):
        self.lmdb_file = lmdb_file
        self.s1_transform = s1_transform
        self.s2a_transform = s2a_transform
        self.s2c_transform = s2c_transform
        self.is_slurm_job = is_slurm_job
        self.subset = subset
        self.normalize = normalize
        self.mode = mode
        self.dtype = dtype

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start since we don't have LMDB at initialization time
            self.env = None
            if self.subset is not None:
                self.length = 50000
            else:
                self.length = 250000

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()

        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())
        
        ## s1
        if self.mode==['s1']:
            s1_bytes, s1_shape = pickle.loads(data)
            if self.dtype=='uint8':
                sample_s1 = np.frombuffer(s1_bytes, dtype=np.uint8).reshape(s1_shape)
            else:
                sample_s1 = np.frombuffer(s1_bytes, dtype=np.float32).reshape(s1_shape)
            if self.s1_transform is not None:
                sample_s1 = self.s1_transform(sample_s1)
            return sample_s1

        ## s2a
        if self.mode==['s2a']:
            s2a_bytes, s2a_shape = pickle.loads(data)
            if self.dtype=='uint8':
                sample_s2a = np.frombuffer(s2a_bytes, dtype=np.uint8).reshape(s2a_shape)
            else:
                sample_s2a = np.frombuffer(s2a_bytes, dtype=np.int16).reshape(s2a_shape)
                sample_s2a = (sample_s2a / 10000.0).astype(np.float32)
            if self.s2a_transform is not None:
                sample_s2a = self.s2a_transform(sample_s2a)
            return sample_s2a
        
        ## s2c
        if self.mode==['s2c']:
            s2c_bytes, s2c_shape = pickle.loads(data)
            if self.dtype=='uint8':
                sample_s2c = np.frombuffer(s2c_bytes, dtype=np.uint8).reshape(s2c_shape)
            else:
                sample_s2c = np.frombuffer(s2c_bytes, dtype=np.int16).reshape(s2c_shape)
                sample_s2c = (sample_s2c / 10000.0).astype(np.float32)
            if self.s2c_transform is not None:
                sample_s2c = self.s2c_transform(sample_s2c)
            return sample_s2c
    
        ## s1, s2a, s2c [TBD, for 50k subset experiments]
        if self.mode==['s1','s2a','s2c']:    
            s1_bytes, s1_shape, s2a_bytes, s2a_shape, s2c_bytes, s2c_shape = pickle.loads(data)
            '''
            if self.dtype=='uint8':
                sample_s1 = np.frombuffer(s1_bytes, dtype=np.uint8).reshape(s1_shape)
                sample_s2a = np.frombuffer(s2a_bytes, dtype=np.uint8).reshape(s2a_shape)
                sample_s2c = np.frombuffer(s2c_bytes, dtype=np.uint8).reshape(s2c_shape)
            else:
            '''
            sample_s1 = np.frombuffer(s1_bytes, dtype=np.float32).reshape(s1_shape)
            sample_s2a = np.frombuffer(s2a_bytes, dtype=np.int16).reshape(s2a_shape)
            sample_s2c = np.frombuffer(s2c_bytes, dtype=np.int16).reshape(s2c_shape)

            #sample_s1 = sample_s1.astype(np.float32)
            #sample_s2a = (sample_s2a / 10000.0).astype(np.float32)
            #sample_s2c = (sample_s2c / 10000.0).astype(np.float32)
            
            #if self.dtype=='uint8':
            #    sample_s2c = (sample_s2c * 255).astype(np.uint8)
            
            if self.s1_transform is not None:
                sample_s1 = self.s1_transform(sample_s1)
            if self.s2a_transform is not None:
                sample_s2a = self.s2a_transform(sample_s2a)
            if self.s2c_transform is not None:
                sample_s2c = self.s2c_transform(sample_s2c)                

            return sample_s1, sample_s2a, sample_s2c
            #return sample_s2c
    
    
    def __len__(self):
        return self.length

    
    

    
    
if __name__ == '__main__':

    
    from cvtorchvision import cvtransforms
    import numpy as np
    import torch
    import random
    from PIL import ImageFilter
    import random
    import cv2

    class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform, season='fixed'):
            self.base_transform = base_transform
            self.season = season

        def __call__(self, x):

            if self.season=='augment':
                season1 = np.random.choice([0,1,2,3])
                season2 = np.random.choice([0,1,2,3])
            elif self.season=='fixed':
                np.random.seed(42)
                season1 = np.random.choice([0,1,2,3])
                season2 = season1
            elif self.season=='random':
                season1 = np.random.choice([0,1,2,3])
                season2 = season1
                
            x1 = np.transpose(x[season1,:,:,:],(1,2,0))
            x2 = np.transpose(x[season2,:,:,:],(1,2,0))

            q = self.base_transform(x1)
            k = self.base_transform(x2)

            return [q, k]


    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            #return x
            return cv2.GaussianBlur(x,(0,0),sigma)


    class RandomBrightness(object):
        """ Random Brightness """

        def __init__(self, brightness=0.4):
            self.brightness = brightness

        def __call__(self, sample):
            s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = sample * s

            return img

    class RandomContrast(object):
        """ Random Contrast """

        def __init__(self, contrast=0.4):
            self.contrast = contrast

        def __call__(self, sample):
            s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = np.mean(sample, axis=(0, 1))

            return ((sample - mean) * s + mean)

    class ToGray(object):
        def __init__(self, out_channels):
            self.out_channels = out_channels
        def __call__(self,sample):
            gray_img = np.mean(sample, axis=-1)
            gray_img = np.tile(gray_img, (self.out_channels, 1, 1))
            gray_img = np.transpose(gray_img, [1, 2, 0])
            return gray_img


    class RandomChannelDrop(object):
        """ Random Channel Drop """

        def __init__(self, min_n_drop=1, max_n_drop=8):
            self.min_n_drop = min_n_drop
            self.max_n_drop = max_n_drop

        def __call__(self, sample):
            n_channels = random.randint(self.min_n_drop, self.max_n_drop)
            channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

            for c in channels:
                sample[c, :, :] = 0        
            return sample

    
    train_transforms_s1 = cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(112, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(2)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),
        #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),        
        cvtransforms.ToTensor()])
    train_transforms_s2a = cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(112, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(12)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),        
        cvtransforms.ToTensor()])
    train_transforms_s2c = cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(112, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(13)], p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),        
        cvtransforms.ToTensor()
    ])
    
    
    train_dataset = LMDBDataset(
        #lmdb_file='/p/scratch/hai_dm4eo/wang_yi/data/ssl4eo_50k.lmdb',
        lmdb_file='/p/scratch/hai_ssl4eo/data/ssl4eo_s12/ssl4eo_250k_s2c_uint8.lmdb',
        #s1_transform=TwoCropsTransform(train_transforms_s1),
        #s2a_transform=TwoCropsTransform(train_transforms_s2a),
        s2c_transform=TwoCropsTransform(train_transforms_s2c,season='augment'),
        is_slurm_job=False,
        #subset=True,
        normalize = False,
        mode = ['s2c'],
        dtype='uint8'
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0)
    print(len(train_dataset))
    for i, (s2c) in enumerate(train_loader):
        if i>1:
            break
        #print(s1[0].shape,s1[0].dtype, s2a[1].shape, s2a[1].dtype, s2c[0].shape,s2c[1].dtype)
        print(s2c[0].shape,s2c[0].dtype,s2c[0].mean(), s2c[1].shape,s2c[1].dtype,s2c[1].mean())

