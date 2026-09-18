import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class AptosGradingDataset(Dataset):
    def __init__(self, split_name, dataset_root, img_size=(224, 224), augment=False, config=None):
        """
        split_name: 'train', 'val', or 'test'
        dataset_root: Path to the raw_aptos directory
        """
        self.split_name = split_name
        self.img_size = img_size
        self.augment = augment
        self.dataset_root = dataset_root
        
        split_mapping = {
            'train': ('train_1.csv', 'train_images/train_images'),
            'val': ('valid.csv', 'val_images/val_images'),
            'test': ('test.csv', 'test_images/test_images')
        }
        
        if split_name not in split_mapping:
            raise ValueError(f"Invalid split_name: {split_name}")
            
        csv_name, img_dir_name = split_mapping[split_name]
        csv_path = os.path.join(dataset_root, csv_name)
        self.img_dir = os.path.join(dataset_root, img_dir_name)
        
        # Load labels
        df = pd.read_csv(csv_path)
        id_col = 'id_code' if 'id_code' in df.columns else df.columns[0]
        label_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[1]
        
        # Set index for easy lookup
        self.df = df.set_index(id_col)
        self.image_names = list(self.df.index)
        self.label_col = label_col
        
        # Define transforms
        if augment and split_name == 'train':
            augment_type = config.get('data', {}).get('augment_type', 'default') if config else 'default'
            if augment_type == 'robust':
                self.transform = T.Compose([
                    T.Resize(img_size),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(img_size),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
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
        
        # APTOS images are typically .png
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        if not os.path.exists(img_path):
            # Fallback to .jpg if .png doesn't exist
            img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[img_name, self.label_col]
        
        if self.transform:
            image = self.transform(image)
            
        return image, int(label), img_name
