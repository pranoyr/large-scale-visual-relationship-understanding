from datasets.vrd import VRDDataset
from datasets.custom_vrd import CustomDataset


def get_training_data(cfg):
    if cfg.DATASET == 'VRD':
        training_data = VRDDataset(cfg.DATASET_DIR, 'train')
    else:
        training_data = CustomDataset(cfg.DATASET_DIR, 'train')

    return training_data


def get_validation_data(cfg):
    if cfg.DATASET == 'VRD':
        validation_data = VRDDataset(cfg.DATASET_DIR, 'test')
    else:
        validation_data = CustomDataset(cfg.DATASET_DIR, 'test')

    return validation_data
