import numpy as np
import torch
import random



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


