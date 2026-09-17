import os
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

class IDRiDGradingDataset(Dataset):
    def __init__(self, split_name, split_json_path, img_dir_train, img_dir_test, labels_train_csv, labels_test_csv, img_size=(224, 224), augment=False):
        self.split_name = split_name
        self.img_size = img_size
        self.augment = augment
        
        with open(split_json_path, 'r') as f:
            splits = json.load(f)
            
        self.image_names = splits[split_name]
        
        df_train = pd.read_csv(labels_train_csv).dropna(how='all', axis=1)
        df_test = pd.read_csv(labels_test_csv).dropna(how='all', axis=1)
        self.df_train = df_train.set_index('Image name')
        self.df_test = df_test.set_index('Image name')
        
        self.img_dirs = {
            'train': img_dir_train,
            'val': img_dir_train,
            'test': img_dir_test
        }
        self.current_img_dir = self.img_dirs[split_name]
        
        if augment and split_name == 'train':
            self.transform = T.Compose([
                T.Resize(img_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.current_img_dir, f"{img_name}.jpg")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert('RGB')
        if self.split_name in ['train', 'val']:
            label = self.df_train.loc[img_name, 'Retinopathy grade']
        else:
            label = self.df_test.loc[img_name, 'Retinopathy grade']
        
        if self.transform:
            image = self.transform(image)
            
        return image, int(label), img_name
