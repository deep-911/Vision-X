import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class IDRiDSegmentationDataset(Dataset):
    def __init__(self, data_root, split, split_json_path, patch_size=(512, 512), augment=False, mode='train'):
        """
        mode: 'train' (random cropping), 'val' (returns full image for sliding window), 'test' (same as val)
        """
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment
        
        self.img_dir = os.path.join(data_root, '1. Original Images', 'a. Training Set' if split in ['train', 'val'] else 'b. Testing Set')
        self.mask_dir = os.path.join(data_root, '2. All Segmentation Groundtruths', 'a. Training Set' if split in ['train', 'val'] else 'b. Testing Set')
        
        with open(split_json_path, 'r') as f:
            splits = json.load(f)
            
        if split == 'test':
            # test is all files in testing set
            self.img_names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(self.img_dir, '*.jpg'))]
        else:
            self.img_names = splits[split]
            
        # Define paths and presence mapping
        self.classes = ['1. Microaneurysms', '2. Haemorrhages', '3. Hard Exudates', '4. Soft Exudates']
        self.class_suffixes = ['_MA.tif', '_HE.tif', '_EX.tif', '_SE.tif']
        
        # Base transforms (normalization)
        self.base_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Medical augmentations
        if self.augment and self.mode == 'train':
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.img_names)
        
    def _load_mask(self, base_name, img_shape):
        masks = []
        validity = []
        for cls_dir, suffix in zip(self.classes, self.class_suffixes):
            mask_path = os.path.join(self.mask_dir, cls_dir, base_name + suffix)
            if os.path.exists(mask_path):
                # Load mask
                m = np.array(Image.open(mask_path))
                if len(m.shape) > 2:
                    m = m[:, :, 0] # sometimes they are saved as RGB
                m = (m > 0).astype(np.float32)
                masks.append(m)
                validity.append(1.0)
            else:
                # Missing mask -> ignore
                masks.append(np.zeros(img_shape, dtype=np.float32))
                validity.append(0.0)
        return np.stack(masks, axis=-1), np.array(validity, dtype=np.float32)

    def __getitem__(self, idx):
        base_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, base_name + '.jpg')
        
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]
        
        mask, validity = self._load_mask(base_name, (h, w))
        
        if self.mode == 'train':
            # Patch sampling logic: 80% lesion-aware, 20% random
            if random.random() < 0.8 and np.sum(validity) > 0:
                # Find available lesion classes
                available_classes = np.where(validity > 0)[0]
                # Pick a random class to focus on
                target_cls = random.choice(available_classes)
                # Find positive pixels
                pos_pixels = np.argwhere(mask[:, :, target_cls] > 0)
                if len(pos_pixels) > 0:
                    center_y, center_x = random.choice(pos_pixels)
                    # Create patch around center with random offset
                    y_min = max(0, center_y - random.randint(0, self.patch_size[0]))
                    x_min = max(0, center_x - random.randint(0, self.patch_size[1]))
                    
                    # Bound checks
                    y_min = min(y_min, h - self.patch_size[0])
                    x_min = min(x_min, w - self.patch_size[1])
                    
                    y_max = y_min + self.patch_size[0]
                    x_max = x_min + self.patch_size[1]
                    
                    img = img[y_min:y_max, x_min:x_max]
                    mask = mask[y_min:y_max, x_min:x_max]
                else:
                    # Fallback to random crop
                    y_min = random.randint(0, h - self.patch_size[0])
                    x_min = random.randint(0, w - self.patch_size[1])
                    img = img[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                    mask = mask[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
            else:
                # 20% Random Background Crop
                y_min = random.randint(0, h - self.patch_size[0])
                x_min = random.randint(0, w - self.patch_size[1])
                img = img[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                mask = mask[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                
            if self.aug_transform:
                augmented = self.aug_transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                
        # Base transform (normalize + to tensor)
        # Transpose mask for albumentations compatibility if needed, but albumentations handles mask
        if self.mode != 'train':
            # full image
            augmented = self.base_transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            augmented = self.base_transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # Mask shape should be [C, H, W]
        mask = mask.permute(2, 0, 1)

        return img, mask, validity, base_name
