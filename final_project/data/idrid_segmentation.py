import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

class IDRiDSegmentationDataset(Dataset):
    def __init__(self, data_root, split, split_json_path, patch_size=(512, 512), augment=False, mode='train', sampling_strategy='lesion_aware'):
        """
        mode: 'train' (random cropping), 'val' (returns full image for sliding window), 'test' (same as val)
        sampling_strategy: 'lesion_aware' (80% any lesion, 20% random) or 'ma_focused' (60% MA, 20% other, 20% random)
        """
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment
        self.sampling_strategy = sampling_strategy
        
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
            
        # Precompute valid images per class for targeted sampling
        self.valid_images_per_class = {0: [], 1: [], 2: [], 3: []}
        if self.mode == 'train':
            for img_name in self.img_names:
                for c, cls_dir, suffix in zip(range(4), self.classes, self.class_suffixes):
                    mask_path = os.path.join(self.mask_dir, cls_dir, img_name + suffix)
                    if os.path.exists(mask_path):
                        # Cheap check: just assume if it exists it has pixels
                        self.valid_images_per_class[c].append(img_name)

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
            if self.sampling_strategy == 'ma_focused':
                r = random.random()
                if r < 0.6:
                    target_cls = 0
                elif r < 0.8:
                    target_cls = random.choice([1, 2, 3])
                else:
                    target_cls = -1
                    
                # If target_cls is chosen, we may need to load a different image if current one doesn't have it
                if target_cls != -1:
                    # check if current image has target_cls
                    has_pixels = (validity[target_cls] > 0 and len(np.argwhere(mask[:, :, target_cls] > 0)) > 0)
                    if not has_pixels:
                        # sample a new image that has this class
                        if len(self.valid_images_per_class[target_cls]) > 0:
                            while True:
                                new_name = random.choice(self.valid_images_per_class[target_cls])
                                new_img_path = os.path.join(self.img_dir, new_name + '.jpg')
                                new_img = np.array(Image.open(new_img_path))
                                new_h, new_w = new_img.shape[:2]
                                new_mask, new_validity = self._load_mask(new_name, (new_h, new_w))
                                
                                pos_pixels = np.argwhere(new_mask[:, :, target_cls] > 0)
                                if len(pos_pixels) > 0:
                                    img, mask, validity, base_name = new_img, new_mask, new_validity, new_name
                                    h, w = new_h, new_w
                                    break
                                else:
                                    # False positive in fast precompute (file exists but no pixels)
                                    self.valid_images_per_class[target_cls].remove(new_name)
                                    if len(self.valid_images_per_class[target_cls]) == 0:
                                        target_cls = -1
                                        break
                        else:
                            target_cls = -1
                            
                # Perform the crop
                if target_cls != -1:
                    pos_pixels = np.argwhere(mask[:, :, target_cls] > 0)
                    if len(pos_pixels) > 0:
                        center_y, center_x = random.choice(pos_pixels)
                        y_min = max(0, center_y - random.randint(0, self.patch_size[0]))
                        x_min = max(0, center_x - random.randint(0, self.patch_size[1]))
                        y_min = min(y_min, h - self.patch_size[0])
                        x_min = min(x_min, w - self.patch_size[1])
                        img = img[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                        mask = mask[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                    else:
                        target_cls = -1 # fallback
                        
                if target_cls == -1: # random background fallback
                    y_min = random.randint(0, h - self.patch_size[0])
                    x_min = random.randint(0, w - self.patch_size[1])
                    img = img[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                    mask = mask[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                    
            elif self.sampling_strategy == 'ma_magnified':
                # 80% MA, 20% random background
                # Native crop size is 256x256
                native_crop = 256
                target_cls = 0 if random.random() < 0.8 else -1
                
                if target_cls == 0:
                    has_pixels = (validity[0] > 0 and len(np.argwhere(mask[:, :, 0] > 0)) > 0)
                    if not has_pixels:
                        if len(self.valid_images_per_class[0]) > 0:
                            while True:
                                new_name = random.choice(self.valid_images_per_class[0])
                                new_img_path = os.path.join(self.img_dir, new_name + '.jpg')
                                new_img = np.array(Image.open(new_img_path))
                                new_h, new_w = new_img.shape[:2]
                                new_mask, new_validity = self._load_mask(new_name, (new_h, new_w))
                                
                                pos_pixels = np.argwhere(new_mask[:, :, 0] > 0)
                                if len(pos_pixels) > 0:
                                    img, mask, validity, base_name = new_img, new_mask, new_validity, new_name
                                    h, w = new_h, new_w
                                    break
                                else:
                                    self.valid_images_per_class[0].remove(new_name)
                                    if len(self.valid_images_per_class[0]) == 0:
                                        target_cls = -1
                                        break
                        else:
                            target_cls = -1
                            
                if target_cls == 0:
                    pos_pixels = np.argwhere(mask[:, :, 0] > 0)
                    if len(pos_pixels) > 0:
                        center_y, center_x = random.choice(pos_pixels)
                        y_min = max(0, center_y - random.randint(0, native_crop))
                        x_min = max(0, center_x - random.randint(0, native_crop))
                        y_min = min(y_min, h - native_crop)
                        x_min = min(x_min, w - native_crop)
                        img = img[y_min:y_min+native_crop, x_min:x_min+native_crop]
                        mask = mask[y_min:y_min+native_crop, x_min:x_min+native_crop]
                    else:
                        target_cls = -1
                        
                if target_cls == -1: # random fallback
                    y_min = random.randint(0, h - native_crop)
                    x_min = random.randint(0, w - native_crop)
                    img = img[y_min:y_min+native_crop, x_min:x_min+native_crop]
                    mask = mask[y_min:y_min+native_crop, x_min:x_min+native_crop]
                    
                # Resize from 256x256 to 512x512 (patch_size)
                # Image uses INTER_LINEAR, mask uses NEAREST
                img = cv2.resize(img, (self.patch_size[1], self.patch_size[0]), interpolation=cv2.INTER_LINEAR)
                # mask is multi-channel [H, W, C]
                mask_resized = np.zeros((self.patch_size[0], self.patch_size[1], 4), dtype=np.float32)
                for c in range(4):
                    mask_resized[:, :, c] = cv2.resize(mask[:, :, c], (self.patch_size[1], self.patch_size[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask_resized
                    
            else: # lesion_aware 80/20
                if random.random() < 0.8 and np.sum(validity) > 0:
                    available_classes = np.where(validity > 0)[0]
                    target_cls = random.choice(available_classes)
                    pos_pixels = np.argwhere(mask[:, :, target_cls] > 0)
                    if len(pos_pixels) > 0:
                        center_y, center_x = random.choice(pos_pixels)
                        y_min = max(0, center_y - random.randint(0, self.patch_size[0]))
                        x_min = max(0, center_x - random.randint(0, self.patch_size[1]))
                        y_min = min(y_min, h - self.patch_size[0])
                        x_min = min(x_min, w - self.patch_size[1])
                        img = img[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                        mask = mask[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                    else:
                        y_min = random.randint(0, h - self.patch_size[0])
                        x_min = random.randint(0, w - self.patch_size[1])
                        img = img[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                        mask = mask[y_min:y_min+self.patch_size[0], x_min:x_min+self.patch_size[1]]
                else:
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
