import numpy as np
import torch
import random
import cv2


class RandomBrightness(object):
    """ Random Brightness """
    
    def __init__(self, brightness=0.4):
        self.brightness = brightness

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        img = sample * s
        
        return img.astype(np.uint8)
    
class RandomContrast(object):
    """ Random Contrast """
    
    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = np.mean(sample, axis=(0, 1))
        
        return ((sample - mean) * s + mean).astype(np.uint8)
    
class ToGray(object):
    def __init__(self, out_channels):
        self.out_channels = out_channels
    def __call__(self,sample):
        gray_img = np.mean(sample, axis=-1)
        gray_img = np.tile(gray_img, (self.out_channels, 1, 1))
        gray_img = np.transpose(gray_img, [1, 2, 0])
        return gray_img.astype(np.uint8)
        
        
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


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        #return x
        return cv2.GaussianBlur(x,(0,0),sigma)
        
        
class Solarize(object):

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def __call__(self, x):
        x1 = x.copy()          
        one = np.ones(x.shape) * 255
        x1[x<self.threshold] = one[x<self.threshold] - x[x<self.threshold]
        
        return x1.astype(np.uint8)
        
class RandomSensorDrop_S1S2(object):
    """ Random Channel Drop """
    
    def __init__(self):
        pass

    def __call__(self, sample):
        sensor = np.random.choice([1,2], replace=False)

        if sensor==2:
            sample[:13, :, :] = 0
        elif sensor==1:
            sample[13:,:,:] = 0
        
        return sample
    
class SensorDrop_S1S2(object):
    def __init__(self, sensor):
        self.sensor = sensor
    def __call__(self,sample):
        if self.sensor == 'S1':
            sample[13:,:,:] = 0
        elif self.sensor == 'S2':
            sample[:13,:,:] = 0
        return sample
    
    
class RandomSensorDrop_RGBD(object):
    """ Random Channel Drop """
    
    def __init__(self):
        pass

    def __call__(self, sample):
        sensor = np.random.choice([1,2], replace=False, p=[0.8,0.2])

        if sensor==2:
            sample[:3, :, :] = 0
        elif sensor==1:
            sample[3:,:,:] = 0
        
        return sample
    
class SensorDrop_RGBD(object):
    def __init__(self, sensor):
        self.sensor = sensor
    def __call__(self,sample):
        if self.sensor == 'D':
            sample[3:,:,:] = 0
        elif self.sensor == 'RGB':
            sample[:3,:,:] = 0
        return sample