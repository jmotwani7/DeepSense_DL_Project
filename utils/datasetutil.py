import json
import random
from pathlib import Path

import h5py
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from augmentations import augmentations_new


class NyuDatasetLoader(Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0)
        dpt = self.dpts[img_idx].transpose(1, 0)

        input_transform = transforms.Compose([augmentations_new.Scale(228),
                                              augmentations_new.ArrayToTensor()])
        target_depth_transform = transforms.Compose([augmentations_new.Scale_Single(228),
                                                     augmentations_new.ArrayToTensor()])

        img = input_transform(img) / 255.
        dpt = target_depth_transform(dpt)

        return img, dpt

    def __len__(self):
        return len(self.lists)


def load_test_train_ids(file_path):
    with open(file_path, 'r') as f:
        ids = json.loads(f.read())
        return ids['train'], ids['val'], ids['test']


def save_test_train_ids(file_path, train_percent=0.8, last_id=1448):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(exist_ok=True)
    all_ids = [i for i in range(last_id)]
    train_ids = random.sample(all_ids, int(last_id * train_percent))

    test_val_ids = [i for i in all_ids if i not in train_ids]

    test_ids = random.sample(test_val_ids, len(test_val_ids) // 2)
    val_ids = [i for i in test_val_ids if i not in test_ids]

    ids_dict = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    with open(file_path, 'w') as f:
        f.write(json.dumps(ids_dict))


def get_nyuv2_test_train_dataloaders(dataset_path, train_ids, val_ids, test_ids, batch_size=32):
    return DataLoader(NyuDatasetLoader(dataset_path, train_ids), batch_size, shuffle=True), \
           DataLoader(NyuDatasetLoader(dataset_path, val_ids), batch_size, shuffle=True), \
           DataLoader(NyuDatasetLoader(dataset_path, test_ids), batch_size, shuffle=True)
