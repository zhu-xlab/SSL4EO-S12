'''
1st band: B2
2nd band: B3
3rd band: B4
4th band: B5
5th band: B6
6th band: B7
7th band: B8
8th band: B8A
9th band: B11 SWIR 
10th band: B12 SWIR 
'''

'''
# random-s2
# train_mean: 0.12428657 0.11001677 0.10230652 0.11532196 0.15989486 0.18204406 0.17513563 0.19565547 0.15648723 0.11122536
# train_std: 0.03922695 0.04709167 0.06532641 0.06240567 0.07583675 0.08917173 0.09050922 0.09968561 0.09901879 0.08733859
# test_mean: 0.12442092 0.11015312 0.10251723 0.11548233 0.15986304 0.18194778 0.17501332 0.1955161  0.15673767 0.11150956
# test_std: 0.03953428 0.04735791 0.06560843 0.06269053 0.07596647 0.08922532 0.09054327 0.09972861 0.09930292 0.08763757

# random-s1
# train_mean: -5.54116458e-05 -1.36324545e-05  4.55894328e-05  2.99090794e-05  4.45195163e-02  2.58623101e-01  3.27207311e-04  1.23416595e-03
# train_std: 0.17569145 0.17611901 0.46005891 0.45636015 2.24921791 7.90565026 2.19176328 1.314848
# test_mean: -2.76615797e-05  2.21397241e-05  2.30798901e-05 -2.89706773e-05  4.52720529e-02  2.66101701e-01  1.96855076e-03  1.48948277e-03
# test_std: 0.17874631 0.17799905 0.46587865 0.46137239 4.02737314 8.28959389 2.79746495 1.67304296


# culture10-s2
# train_mean: 0.12375696 0.10927746 0.10108552 0.11423986 0.15926567 0.18147236 0.17457403 0.19501607 0.15428469 0.10905051
# train_std: 0.03958796 0.04777826 0.06636617 0.06358875 0.07744387 0.09101635 0.09218467 0.10164581 0.09991773 0.08780633
# val_mean: 0.12897079 0.1163941  0.11214067 0.1239297  0.16454521 0.18617419 0.17903959 0.20023431 0.17357621 0.12763362
# val_std: 0.03836618 0.04338256 0.05851725 0.05473533 0.06438882 0.07570399 0.07841136 0.08541158 0.09387924 0.08418835
# test_mean: 0.12777465 0.11487813 0.11098373 0.12303222 0.1643186  0.18593616 0.17902002 0.19994389 0.17236035 0.12748551
# test_std: 0.03511035 0.04017856 0.05515728 0.05085581 0.0614933 0.0729276 0.07586886 0.08249681 0.08809429 0.08089951

# culture10-s1
# train_mean: -3.59122426e-05 -7.65856128e-06 5.93738575e-05 2.51662315e-05 4.42011066e-02 2.57610271e-01 7.55674337e-04 1.35034668e-03
# train_std: 0.17555201 0.17556463 0.45998793 0.45598876 2.85599092 8.32480061 2.44987574 1.4647353
# val_mean: -1.49371637e-04 -5.46514151e-05 -1.26352906e-04 1.72238483e-04 4.69830197e-02 2.72650123e-01 1.23383006e-04 2.51766991e-04
# val_std: 0.18061702 0.18096459 0.4656792 0.46126013 0.77899222 5.25347128 0.561716 0.5400661
# test_mean: -1.54123483e-04 5.84000367e-05 -5.81174764e-05 -2.37403796e-04 4.91872306e-02 2.84092939e-01 -2.89945003e-04 1.36458698e-03
# test_std: 0.18277671 0.18528839 0.47456981 0.47319214 1.11181292 4.32818594 1.34458178 0.79224516
'''

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from cvtorchvision import cvtransforms


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


'''
with h5py.File('culture_10/training.h5','r') as f:
    #s1 = np.array(f['sen1'])
    s2 = np.array(f['sen2'])
    length = s2.shape[0]

max_q = np.quantile(s2.reshape(length*32*32,10),0.98,axis=0)
min_q = np.quantile(s2.reshape(length*32*32,10),0.02,axis=0)
'''

# B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
SO2SAT_STD = [0.03928846, 0.04714491, 0.06538279, 0.0624626, 0.07586263, 0.08918243, 0.09051602, 0.0996942, 0.09907556, 0.08739836]
SO2SAT_MEAN = [0.12431336, 0.11004396, 0.10234854, 0.11535393, 0.15988852, 0.18202487, 0.17511124, 0.19562768, 0.15653716, 0.11128203]



def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img




class So2SatDataset(Dataset):
    def __init__(self,path,bands=None,quantile=None,transform=None, normalize=None):
        self.path = path
        self.bands = bands
        self.transform = transform       
        self.quantile = quantile
        self.normalize = normalize
        with h5py.File(self.path,'r') as f:
            self.length = f['label'].shape[0]
        
        #self.mean = [0.12431336, 0.11004396, 0.10234854, 0.11535393, 0.15988852, 0.18202487, 0.17511124, 0.19562768, 0.15653716, 0.11128203]
        #self.std = [0.03928846, 0.04714491, 0.06538279, 0.0624626, 0.07586263, 0.08918243, 0.09051602, 0.0996942, 0.09907556, 0.08739836]
       
    def __getitem__(self, index):
        
        with h5py.File(self.path,'r') as f:
            patch = f['sen2'][index]
            label = f['label'][index]

        
        channels = []
        for b in range(10):
            ch = patch[:,:,b]
            if self.normalize:
                ch = normalize(patch[:,:,b],SO2SAT_MEAN[b],SO2SAT_STD[b])
            channels.append(ch)
        img = np.dstack(channels)
        

        if self.bands == 'RGB':
            sample = img[:,:,0:-1:3]
        elif self.bands == 'B12':            
            sample = np.concatenate((np.zeros((32,32,1),dtype=np.uint8),img[:,:,0:8],np.zeros((32,32,1),dtype=np.uint8),img[:,:,-2:]),axis=-1)
        elif self.bands == 'B13':
            sample = np.concatenate((np.zeros((32,32,1),dtype=np.uint8),img[:,:,0:8],np.zeros((32,32,2),dtype=np.uint8),img[:,:,-2:]),axis=-1)
        else:
            sample = img
        '''    
        if self.quantile is not None:
            self.max_q = quantile[1]
            self.min_q = quantile[0]
            img_bands = []
            for b in range(10):
                img = sample[:,:,b]
                
                max_q = self.max_q[b]
                min_q = self.min_q[b]
                img[img>max_q] = max_q
                img[img<min_q] = min_q
                img = img.reshape(32,32,1)
                img_bands.append(img)
            sample = np.concatenate(img_bands,axis=2)
        '''
        sample = (sample*255).astype(np.uint8)
        target = label.astype(np.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.length









if __name__ == '__main__':


    train_path = 'culture_10/training.h5'
    val_path = 'culture_10/testing.h5'

    batchsize = 4

    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(32),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor(),
            ])

    val_transforms = cvtransforms.Compose([
            #cvtransforms.Resize(32),
            #cvtransforms.CenterCrop(32),
            cvtransforms.ToTensor(),
            ])

    train_dataset = So2SatDataset(train_path,bands=None,transform=train_transforms)
    val_dataset = So2SatDataset(val_path,bands=None,transform=val_transforms)

    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True,num_workers=2,pin_memory=False,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=False,drop_last=True)

    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))


