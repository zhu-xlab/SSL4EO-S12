import random

from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from pytorch_lightning import LightningDataModule

from datasets.oscd_dataset import ChangeDetectionDataset


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
S2A_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']
BGR_BANDS = ['B02', 'B03', 'B04']

class RandomFlip:

    def __call__(self, *xs):
        if random.random() > 0.5:
            xs = tuple(TF.hflip(x) for x in xs)
        return xs


class RandomRotation:

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, *xs):
        angle = random.choice(self.angles)
        return tuple(TF.rotate(x, angle) for x in xs)


class RandomSwap:

    def __call__(self, x1, x2, y):
        if random.random() > 0.5:
            return x2, x1, y
        else:
            return x1, x2, y


class ToTensor:

    def __call__(self, *xs):
        return tuple(TF.to_tensor(x) for x in xs)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *xs):
        for t in self.transforms:
            xs = t(*xs)
        return xs


class ChangeDetectionDataModule(LightningDataModule):

    def __init__(self, data_dir, RGB_bands=True, BGR_bands=False, S2A_bands=False,
                 value_discard=True, patch_size=96, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.value_discard = value_discard
        if RGB_bands:
            if BGR_bands:
                self.bands = BGR_BANDS 
            else:
                self.bands = RGB_BANDS
        elif S2A_bands:
            self.bands = S2A_BANDS
        else:
            self.bands = ALL_BANDS
            

    def setup(self, stage=None):
        self.train_dataset = ChangeDetectionDataset(
            self.data_dir,
            split='train',
            bands=self.bands,
            value_discard=self.value_discard,
            transform=Compose([ToTensor(), RandomFlip(), RandomRotation()]), # here need first call ToTensor to convert np.array to tensors in order to do following transformations
            patch_size=self.patch_size
        )
        self.val_dataset = ChangeDetectionDataset(
            self.data_dir,
            split='test',
            bands=self.bands,
            value_discard=self.value_discard,
            transform=ToTensor(),
            patch_size=self.patch_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=274,
            shuffle=False, # True,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )
