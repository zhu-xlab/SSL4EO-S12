# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import torch
import numpy as np

from torchvision import datasets, transforms


from timm.data import create_transform

from .dall_e.utils import map_pixels
from .masking_generator import MaskingGenerator
from .dataset_folder import ImageFolder
from PIL import Image
from models.moco_v3 import loader as moco_loader
from datasets.SSL4EO.ssl4eo_dataset_lmdb import LMDBDataset
from sklearn.model_selection import train_test_split


from cvtorchvision import cvtransforms



class SeasonTransform:

    def __init__(self, base_transform, season='fixed'):
        self.base_transform = base_transform
        self.season = season

    def __call__(self, x):

        if self.season=='augment':
            season1 = np.random.choice([0,1,2,3])
            season2 = np.random.choice([0,1,2,3])
            
            x1 = np.transpose(x[season1,:,:,:],(1,2,0))
            x2 = np.transpose(x[season2,:,:,:],(1,2,0))            
            image = self.base_transform(x1)
            #target = self.base_transform2(x2)
            return image, target
            
        elif self.season=='fixed':
            np.random.seed(42)
            season1 = np.random.choice([0,1,2,3])

        elif self.season=='random':
            season1 = np.random.choice([0,1,2,3])

        x1 = np.transpose(x[season1,:,:,:],(1,2,0))
        image = self.base_transform(x1)
        return image

class DataAugmentations(object):
    def __init__(self, args):
        if args.aug_level == 0:
            print("Please")
            base_transform = transforms.Compose([
                cvtransforms.CenterCrop(size=args.in_size), 
            ])
            
        elif args.aug_level == 1:
            base_transform = transforms.Compose([
                cvtransforms.CenterCrop(size=args.in_size), 
                cvtransforms.RandomHorizontalFlip()
            ])
        elif args.aug_level == 2:
            base_transform = transforms.Compose([
                cvtransforms.RandomResizedCrop(args.in_size, scale=(args.crop_min, 1.)), 
                cvtransforms.RandomHorizontalFlip()
            ])
        elif args.aug_level == 3:
            base_transform = transforms.Compose([
                cvtransforms.RandomResizedCrop(args.in_size, scale=(args.crop_min, 1.)),
                cvtransforms.RandomApply([
                    RandomBrightness(0.4),
                    RandomContrast(0.4)
                ], p=0.8),
                cvtransforms.RandomApply([ToGray(13)], p=0.2),
                cvtransforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=1.0),
                cvtransforms.RandomHorizontalFlip(),
                cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),  
            ])  
        else:
            base_transform = transforms.Compose([cvtransforms.ToTensor()])
            
        self.common_transform = SeasonTransform(base_transform, season=args.season)
        
        self.patch_transform = transforms.Compose([
                cvtransforms.ToTensor()
                #transforms.Normalize(
                #    mean=torch.tensor(0),
                #    std=torch.tensor(1))
            ])
        
        if getattr(args, 'discrete_vae_type', None) is None:
            self.visual_token_transform = lambda z: z
        elif args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()
        
        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )
    
    def __call__(self, image):
        z = self.common_transform(image)
        if isinstance(z, tuple):
            for_patches, for_visual_tokens = z
            return \
                self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
                self.masked_position_generator()
        else:
            return self.patch_transform(z), self.masked_position_generator()


    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

    

def build_beit_pretraining_dataset(args):
    transform = DataAugmentations(args)
    train_dataset = LMDBDataset(
        lmdb_file=args.data_path,
        s2c_transform=transform,#TwoCropsTransform(base_transform1=cvtransforms.Compose(augmentation1), base_transform2 = cvtransforms.Compose(augmentation2),season=args.season),
        is_slurm_job=False,#args.is_slurm_job,
        normalize=False,
        dtype=args.dtype,
        mode=args.mode
    )
    return train_dataset
    

        

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
