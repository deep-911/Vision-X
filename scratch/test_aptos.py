from mvp.data.registry import get_dataset
import yaml

def test_aptos():
    config = {
        'data': {
            'dataset_root': r'Datasets/aptos/raw_aptos',
            'img_size': [224, 224]
        }
    }
    
    print("Testing train split...")
    train_dataset = get_dataset('aptos', 'train', config, augment=True)
    print(f"Train size: {len(train_dataset)}")
    img, label, name = train_dataset[0]
    print(f"Sample 0 - Image shape: {img.shape}, Label: {label}, Name: {name}")
    
    print("Testing val split...")
    val_dataset = get_dataset('aptos', 'val', config, augment=False)
    print(f"Val size: {len(val_dataset)}")
    img, label, name = val_dataset[0]
    print(f"Sample 0 - Image shape: {img.shape}, Label: {label}, Name: {name}")

if __name__ == "__main__":
    test_aptos()
