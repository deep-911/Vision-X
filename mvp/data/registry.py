from mvp.data.datasets.idrid_grading import IDRiDGradingDataset
from mvp.data.datasets.aptos_grading import AptosGradingDataset

def get_dataset(dataset_name, split_name, config, augment=False):
    """
    Generic dataset registry to instantiate datasets by name.
    """
    if dataset_name == 'idrid':
        return IDRiDGradingDataset(
            split_name=split_name,
            split_json_path=config['data']['split_json_path'],
            img_dir_train=config['data']['img_dir_train'],
            img_dir_test=config['data']['img_dir_test'],
            labels_train_csv=config['data']['labels_train_csv'],
            labels_test_csv=config['data']['labels_test_csv'],
            img_size=tuple(config['data'].get('img_size', [224, 224])),
            augment=augment
        )
    elif dataset_name == 'aptos':
        return AptosGradingDataset(
            split_name=split_name,
            dataset_root=config['data']['dataset_root'],
            img_size=tuple(config['data'].get('img_size', [224, 224])),
            augment=augment,
            config=config
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
