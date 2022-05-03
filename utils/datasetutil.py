import glob
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from skimage import transform
from torch.utils.data import DataLoader, Dataset

from augmentations import augmentations_new


class CityScapeGenerator(Dataset):
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
            img_dep_transform = augmentations_new.Compose([augmentations_new.Scale(228),
                                                           augmentations_new.ArrayToTensor()])
        img, dpt = img_dep_transform(img, dpt)

        return img / 255., dpt


# class NyuDatasetLoader(Dataset):
#     def __init__(self, data_path, lists, augment_data=True):
#         self.data_path = data_path
#         self.lists = lists
#         self.augment_data = augment_data
#
#         self.nyu = h5py.File(self.data_path)
#
#         self.imgs = self.nyu['images']
#         self.dpts = self.nyu['depths']
#         if self.augment_data:
#             self.img_dep_transform = augmentations_new.Compose([augmentations_new.RandomVerticalFlip(),
#                                                                 augmentations_new.RandomHorizontalFlip(),
#                                                                 augmentations_new.RandomRotate(15),
#                                                                 augmentations_new.AorB(augmentations_new.Scale(228), augmentations_new.RandomCrop((228, 304)), probA=0.8),
#                                                                 augmentations_new.ArrayToTensor()
#                                                                 ])
#         else:
#             self.img_dep_transform = augmentations_new.Compose([augmentations_new.Scale(228),
#                                                                 augmentations_new.ArrayToTensor()])
#
#     def __getitem__(self, index):
#         img_idx = self.lists[index]
#         img = self.imgs[img_idx].transpose(2, 1, 0)
#         dpt = self.dpts[img_idx].transpose(1, 0)
#
#         img, dpt = self.img_dep_transform(img, dpt)
#         return img / 255., dpt
#
#     def __len__(self):
#         return len(self.lists)

class NyuDatasetLoader(Dataset):
    def __init__(self, data_path, lists, augment_data=True, augment_size=100):
        self.data_path = data_path
        self.lists = lists
        self.augment_data = augment_data
        self.augment_size = augment_size

        self.imgs = []  # self.nyu['images']
        self.dpts = []  # self.nyu['depths']
        self.default_transform = augmentations_new.Compose([augmentations_new.Scale(228),
                                                            augmentations_new.ArrayToTensor()])
        if self.augment_data:
            self.augmentation_transform = augmentations_new.Compose([augmentations_new.RandomVerticalFlip(),
                                                                     augmentations_new.RandomHorizontalFlip(),
                                                                     augmentations_new.AorB(augmentations_new.Scale(228),
                                                                                            augmentations_new.AorB(augmentations_new.Compose([augmentations_new.RandomRotate(30), augmentations_new.CenterCrop((228, 304))]),
                                                                                                                   augmentations_new.Compose([augmentations_new.RandomCenterCrop((228, 304)), augmentations_new.ScaleExact((228, 304))])), probA=0.25),
                                                                     augmentations_new.ArrayToTensor()
                                                                     ])
        self.initialize_augmentations(h5py.File(self.data_path))

    def initialize_augmentations(self, nyu_dataset):
        # for img, dep in zip(nyu_dataset['images'], nyu_dataset['depths']):
        for idx in self.lists:
            img = nyu_dataset['images'][idx].transpose(2, 1, 0)
            dep = nyu_dataset['depths'][idx].transpose(1, 0)
            t_img, t_dep = self.default_transform(img, dep)
            self.imgs.append(t_img / 255.)
            self.dpts.append(t_dep)

        if self.augment_data:
            for i in range(self.augment_size):
                rand_idx = random.sample(self.lists, 1)[0]
                img = nyu_dataset['images'][rand_idx].transpose(2, 1, 0)
                dep = nyu_dataset['depths'][rand_idx].transpose(1, 0)
                try:
                    t_img, t_dep = self.augmentation_transform(img, dep)
                    self.imgs.append(t_img / 255.)
                    self.dpts.append(t_dep)
                except:
                    print('Exception occured while applying augmentation transform')
                if i % 500 == 0:
                    print(f'Generated {i} data augmentations')

    def __getitem__(self, index):
        # img_idx = self.lists[index]
        return self.imgs[index], self.dpts[index]

    def __len__(self):
        return len(self.imgs)


# class NyuDatasetLoaderLight(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#
#         self.imgs = []
#         self.dpts = []
#
#         self.default_transform = augmentations_new.ArrayToTensor()
#         npy_dataset = self.load_pickle(self.data_path)
#         for i in range(npy_dataset.shape[0]):
#             img, dep = self.default_transform(npy_dataset[i, :, :, :3], npy_dataset[i, :, :, 3])
#             self.imgs.append(img)
#             self.dpts.append(dep)
#
#     def load_pickle(self, dataset_loc):
#         dataset = np.load(dataset_loc)
#         print(f'Dataset loaded successfully {dataset[0]}')
#         return dataset
#
#     def __getitem__(self, index):
#         return self.imgs[index], self.dpts[index]
#
#     def __len__(self):
#         return len(self.imgs)


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


def get_nyuv2_test_train_dataloaders(dataset_path, train_ids, val_ids, test_ids, batch_size=3, apply_augmentations=True, augmentations_count=5000):
    # return #DataLoader(NyuDatasetLoader(dataset_path, train_ids, augment_data=apply_augmentations, augment_size=augmentations_count), batch_size, shuffle=True),
    return None, DataLoader(NyuDatasetLoader(dataset_path, val_ids, augment_data=apply_augmentations, augment_size=int(augmentations_count * 0.2)), batch_size, shuffle=True), \
           DataLoader(NyuDatasetLoader(dataset_path, test_ids), batch_size, shuffle=True)


# def get_nyuv2_test_train_dataloaders(dataset_path, train_ids, val_ids, test_ids, batch_size=3, apply_augmentations=True, augmentations_count=10000):
#     return DataLoader(NyuDatasetLoaderLight(dataset_path), batch_size, shuffle=True), \
#            DataLoader(NyuDatasetLoaderLight(dataset_path), batch_size, shuffle=True), \
#            DataLoader(NyuDatasetLoaderLight(dataset_path, test_ids), batch_size, shuffle=True)


def get_cityscape_val_train_dataloader(dataset_path, batch_size=32):
    return DataLoader(CityScapeGenerator(dataset_path, "train"), batch_size), \
           DataLoader(CityScapeGenerator(dataset_path, "val"), batch_size)

# train_loader, val_loader = get_cityscape_val_train_dataloader("../datasets/cityscapes")
