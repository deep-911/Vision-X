import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class IDRiDMASpecialistDataset(Dataset):
    def __init__(self, root_dir, split, split_json, patch_size=(512, 512), augment=False, mode='train'):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.mode = mode
        self.native_crop = 256
        
        self.img_dir = os.path.join(root_dir, '1. Original Images', 'a. Training Set')
        self.mask_dir = os.path.join(root_dir, '2. All Segmentation Groundtruths', 'a. Training Set')
        
        with open(split_json, 'r') as f:
            splits = json.load(f)
        self.img_names = splits[split]
        
        # Tracking metrics for audit
        self.stats = {
            'pos_sampled': 0,
            'hard_neg_sampled': 0,
            'neutral_neg_sampled': 0,
            'hard_neg_rejected': 0,
            'neutral_neg_rejected': 0,
            'pos_ma_counts': []
        }
        
        self.transform = None
        if self.augment and mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.2),
            ])
            
        self.to_tensor = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.valid_images_per_class = {c: [] for c in range(4)}
        for name in self.img_names:
            for c, prefix in enumerate(['1. Microaneurysms', '2. Haemorrhages', '3. Hard Exudates', '4. Soft Exudates']):
                mask_path = glob.glob(os.path.join(self.mask_dir, prefix, f"{name}_*.tif"))
                if len(mask_path) > 0:
                    self.valid_images_per_class[c].append(name)
                    
        # For training, precompute MA connected components for faster sampling
        self.ma_components = {}
        if self.mode == 'train':
            print("Precomputing MA components for positive sampling...")
            for name in self.valid_images_per_class[0]:
                mask_path = glob.glob(os.path.join(self.mask_dir, '1. Microaneurysms', f"{name}_*.tif"))[0]
                mask = np.array(Image.open(mask_path))
                if mask.sum() > 0:
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    if num_labels > 1:
                        # Skip background (label 0)
                        self.ma_components[name] = {
                            'centroids': centroids[1:],
                            'stats': stats[1:],
                            'mask': mask
                        }

    def _load_mask(self, img_name, shape):
        h, w = shape
        masks = []
        validity = []
        prefixes = ['1. Microaneurysms', '2. Haemorrhages', '3. Hard Exudates', '4. Soft Exudates']
        for prefix in prefixes:
            mask_path = glob.glob(os.path.join(self.mask_dir, prefix, f"{img_name}_*.tif"))
            if len(mask_path) > 0:
                m = np.array(Image.open(mask_path[0]))
                m = (m > 0).astype(np.float32)
                masks.append(m)
                validity.append(1.0)
            else:
                masks.append(np.zeros((h, w), dtype=np.float32))
                validity.append(0.0)
        return np.stack(masks, axis=-1), np.array(validity, dtype=np.float32)

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_names) * 4 # arbitrary epoch multiplier
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.mode == 'val':
            name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, name + '.jpg')
            img = np.array(Image.open(img_path))
            h, w = img.shape[:2]
            mask, validity = self._load_mask(name, (h, w))
            
            if self.to_tensor:
                augmented = self.to_tensor(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].permute(2, 0, 1) # [4, H, W]
            
            # Return only MA mask and validity for specialist
            return img, mask[0:1], validity[0:1], name
            
        # Training sampling logic
        rnd = random.random()
        if rnd < 0.5:
            target_type = 'pos'
        elif rnd < 0.75:
            target_type = 'hard_neg'
        else:
            target_type = 'neutral_neg'
            
        success = False
        img, mask, validity, base_name = None, None, None, None
        
        # 1. Positive Sampling
        if target_type == 'pos':
            available_imgs = list(self.ma_components.keys())
            if len(available_imgs) > 0:
                name = random.choice(available_imgs)
                img_path = os.path.join(self.img_dir, name + '.jpg')
                img = np.array(Image.open(img_path))
                h, w = img.shape[:2]
                mask, validity = self._load_mask(name, (h, w))
                
                centroids = self.ma_components[name]['centroids']
                cx, cy = random.choice(centroids)
                
                y_min = max(0, int(cy) - random.randint(0, self.native_crop))
                x_min = max(0, int(cx) - random.randint(0, self.native_crop))
                y_min = min(y_min, h - self.native_crop)
                x_min = min(x_min, w - self.native_crop)
                
                img = img[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                mask = mask[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                base_name = name
                
                # Count MAs in this crop
                ma_mask = mask[:, :, 0]
                if ma_mask.sum() > 0:
                    num_l, _ = cv2.connectedComponents(ma_mask.astype(np.uint8), connectivity=8)
                    self.stats['pos_ma_counts'].append(num_l - 1)
                else:
                    self.stats['pos_ma_counts'].append(0)
                    
                self.stats['pos_sampled'] += 1
                success = True
            else:
                target_type = 'neutral_neg'
                
        # 2. Hard Negative Sampling
        if target_type == 'hard_neg':
            available_classes = [c for c in [1, 2, 3] if len(self.valid_images_per_class[c]) > 0]
            if len(available_classes) > 0:
                for _ in range(10): # try 10 times to find a hard negative without MA
                    neg_class = random.choice(available_classes)
                    name = random.choice(self.valid_images_per_class[neg_class])
                    img_path = os.path.join(self.img_dir, name + '.jpg')
                    tmp_img = np.array(Image.open(img_path))
                    h, w = tmp_img.shape[:2]
                    tmp_mask, tmp_validity = self._load_mask(name, (h, w))
                    
                    pos_pixels = np.argwhere(tmp_mask[:, :, neg_class] > 0)
                    if len(pos_pixels) > 0:
                        cy, cx = random.choice(pos_pixels)
                        y_min = max(0, int(cy) - random.randint(0, self.native_crop))
                        x_min = max(0, int(cx) - random.randint(0, self.native_crop))
                        y_min = min(y_min, h - self.native_crop)
                        x_min = min(x_min, w - self.native_crop)
                        
                        crop_mask = tmp_mask[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                        
                        # Verify NO MA pixels in this crop
                        if tmp_validity[0] > 0 and crop_mask[:, :, 0].sum() > 0:
                            self.stats['hard_neg_rejected'] += 1
                            continue # Has MA, reject and resample
                            
                        img = tmp_img[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                        mask = crop_mask
                        validity = tmp_validity
                        base_name = name
                        self.stats['hard_neg_sampled'] += 1
                        success = True
                        break
                        
            if not success:
                target_type = 'neutral_neg'
                
        # 3. Neutral Negative Sampling
        if target_type == 'neutral_neg':
            for _ in range(10):
                name = random.choice(self.img_names)
                img_path = os.path.join(self.img_dir, name + '.jpg')
                tmp_img = np.array(Image.open(img_path))
                h, w = tmp_img.shape[:2]
                tmp_mask, tmp_validity = self._load_mask(name, (h, w))
                
                y_min = random.randint(0, h - self.native_crop)
                x_min = random.randint(0, w - self.native_crop)
                
                crop_mask = tmp_mask[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                
                if tmp_validity[0] > 0 and crop_mask[:, :, 0].sum() > 0:
                    self.stats['neutral_neg_rejected'] += 1
                    continue
                    
                img = tmp_img[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                mask = crop_mask
                validity = tmp_validity
                base_name = name
                self.stats['neutral_neg_sampled'] += 1
                success = True
                break
                
            # Ultimate fallback if we somehow can't find a pure negative patch in 10 tries
            if not success:
                # Just take the last crop, but forcibly zero out the MA mask to ensure negative objective
                # This is a safe fallback since we are building an MA specialist.
                img = tmp_img[y_min:y_min+self.native_crop, x_min:x_min+self.native_crop]
                mask = crop_mask
                mask[:, :, 0] = 0 
                validity = tmp_validity
                base_name = name
                self.stats['neutral_neg_sampled'] += 1

        # Resize from 256x256 to 512x512
        img = cv2.resize(img, (self.patch_size[1], self.patch_size[0]), interpolation=cv2.INTER_LINEAR)
        # mask is [H, W, 4]
        mask_resized = np.zeros((self.patch_size[0], self.patch_size[1], 4), dtype=np.float32)
        for c in range(4):
            mask_resized[:, :, c] = cv2.resize(mask[:, :, c], (self.patch_size[1], self.patch_size[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask_resized
        
        # Only keep channel 0 for MA specialist
        mask = mask[:, :, 0:1] # [H, W, 1]
        validity = validity[0:1] # [1]
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        if self.to_tensor:
            augmented = self.to_tensor(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].permute(2, 0, 1) # [1, H, W]
            
        return img, mask, validity, base_name
