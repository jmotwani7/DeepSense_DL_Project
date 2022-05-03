import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import random
import json
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import os
from augmentations import augmentations_new
from skimage import transform


class CityScapeGenerator(torch.utils.data.Dataset):
    def __init__(self, root, split="train", augment_data=True):
        self.root = root
        self.split = split
        self.augment_data = augment_data
        self.files = {}
        for split in ["train", "val"]:
            file_list = glob.glob(root + "/" + split + "/" + "image/**")
            self.files[split] = file_list

    def __len__(self):
        # return int(np.ceil(len(self.data) / self.batch_size))
        return len(self.files[self.split])

    def __getitem__(self, id):
        img_path = self.files[self.split][id]
        img_name = os.path.split(self.files[self.split][0])[-1][:-4]
        dpt_path = self.root + "/" + self.split + "/" + "depth/" + img_name + ".jpg"
        img = np.asarray(Image.open(img_path))
        dpt = np.asarray(Image.open(dpt_path))
        img = transform.resize(img, (480, 640))
        dpt = transform.resize(dpt, (480, 640))
        if self.augment_data:
            img_dep_transform = augmentations_new.Compose([augmentations_new.RandomVerticalFlip(),
                                                           augmentations_new.RandomHorizontalFlip(),
                                                           augmentations_new.RandomRotate(15),
                                                           augmentations_new.AorB(augmentations_new.Scale(228), augmentations_new.RandomCrop((228, 304)), probA=0.8),
                                                           augmentations_new.ArrayToTensor()
                                                           ])
        else:
            img_dep_transform = transforms.Compose([augmentations_new.Scale(228),
                                                    augmentations_new.ArrayToTensor()])
        img, dpt = img_dep_transform(img, dpt)

        return img / 255., dpt


class NyuDatasetLoader(Dataset):
    def __init__(self, data_path, lists, augment_data=True):
        self.data_path = data_path
        self.lists = lists
        self.augment_data = augment_data

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0)
        dpt = self.dpts[img_idx].transpose(1, 0)

        if self.augment_data:
            img_dep_transform = augmentations_new.Compose([augmentations_new.RandomVerticalFlip(),
                                                           augmentations_new.RandomHorizontalFlip(),
                                                           augmentations_new.RandomRotate(15),
                                                           augmentations_new.AorB(augmentations_new.Scale(228), augmentations_new.RandomCrop((228, 304)), probA=0.8),
                                                           augmentations_new.ArrayToTensor()
                                                           ])
        else:
            img_dep_transform = transforms.Compose([augmentations_new.Scale(228),
                                                    augmentations_new.ArrayToTensor()])

        img, dpt = img_dep_transform(img, dpt)
        return img / 255., dpt

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


def get_cityscape_val_train_dataloader(dataset_path, batch_size=32):
    return DataLoader(CityScapeGenerator(dataset_path, "train"), batch_size), \
           DataLoader(CityScapeGenerator(dataset_path, "val"), batch_size)

# train_loader, val_loader = get_cityscape_val_train_dataloader("../datasets/cityscapes")
