import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from cvtorchvision import cvtransforms
from pathlib import Path
import os
import rasterio
import cv2


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
S2A_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']
RGB_BANDS = ['B04', 'B03', 'B02']

### SSL4EO stats
BAND_STATS = {
    'mean': {
        'B01': 1353.72696296,
        'B02': 1117.20222222,
        'B03': 1041.8842963,
        'B04': 946.554,
        'B05': 1199.18896296,
        'B06': 2003.00696296,
        'B07': 2374.00874074,
        'B08': 2301.22014815,
        'B8A': 2599.78311111,
        'B09': 732.18207407,
        'B10': 12.09952894,
        'B11': 1820.69659259,
        'B12': 1118.20259259
    },
    'std': {
        'B01': 897.27143653,
        'B02': 736.01759721,
        'B03': 684.77615743,
        'B04': 620.02902871,
        'B05': 791.86263829,
        'B06': 1341.28018273,
        'B07': 1595.39989386,
        'B08': 1545.52915718,
        'B8A': 1750.12066835,
        'B09': 475.11595216,
        'B10': 98.26600935,
        'B11': 1216.48651476,
        'B12': 736.6981037
    }
}


def is_valid_file(filename):
    return filename.lower().endswith(EXTENSIONS)

def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class EurosatDataset(Dataset):

    def __init__(self, root, bands='B13', transform=None, normalize=False):
        self.root = Path(root)
        self.transform = transform
        if bands=='B13':
            self.bands = ALL_BANDS
        elif bands=='B12':
            self.bands = S2A_BANDS
        elif bands=='RGB':
            self.bands = RGB_BANDS
            
        self.normalize = normalize
            
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []

        for froot, _, fnames in sorted(os.walk(root, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(froot, fname)
                    self.samples.append(path)
                    target = self.class_to_idx[Path(path).parts[-2]]
                    self.targets.append(target)

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]
        
        with rasterio.open(path) as f:
            if self.bands == ALL_BANDS:
                array = f.read().astype(np.int16)
            elif self.bands == S2A_BANDS:
                array = f.read((1,2,3,4,5,6,7,8,9,11,12,13)).astype(np.int16)
            elif self.bands == RGB_BANDS:
                array = f.read((4,3,2)).astype(np.int16)
                            
            img = array.transpose(1, 2, 0)

        channels = []
        
        for i,b in enumerate(self.bands):
            ch = img[:,:,i]
            if self.normalize:
                ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
            else:
                ch = (ch / 10000.0 * 255.0).astype('uint8')

            if b=='B8A': # EuSAT band order is different than SSL4EO
                channels.insert(8,ch)
            else:
                channels.append(ch)
        img = np.dstack(channels)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, target = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, target

    def __len__(self):
        return len(self.indices)



if __name__ == '__main__':


    data_path = 'eurosat'
    batchsize = 4
    
    eurosat_dataset = EurosatDataset(root='/p/project/hai_dm4eo/wang_yi/data/eurosat/tif')

    from sklearn.model_selection import train_test_split
    indices = np.arange(len(eurosat_dataset))
    train_indices, val_indices = train_test_split(indices, train_size=0.5,stratify=eurosat_dataset.targets)

    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(56),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor(),
            ])

    val_transforms = cvtransforms.Compose([
            #cvtransforms.Resize(64),
            cvtransforms.CenterCrop(56),
            cvtransforms.ToTensor(),
            ])

    train_dataset = Subset(eurosat_dataset, train_indices, train_transforms)
    val_dataset = Subset(eurosat_dataset, val_indices, val_transforms)

    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=False,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=False,drop_last=True)

    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))

    for i, data in enumerate(train_loader):
        if i>10:
            break
        print(data[0].shape,data[1],data[0][0].max())
