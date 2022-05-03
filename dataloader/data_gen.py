import random

import h5py
from torch.utils.data import Dataset
import numpy as np

from augmentations import augmentations_new
from utils.datasetutil import load_test_train_ids


class NyuDatasetLoader(Dataset):
    def __init__(self, data_path, lists, augment_data=True, augment_size=10):
        self.data_path = data_path
        self.lists = lists
        self.augment_data = augment_data
        self.augment_size = augment_size

        self.imgs = []  # self.nyu['images']
        self.dpts = []  # self.nyu['depths']

        if self.augment_data:
            self.augmentation_transform = augmentations_new.Compose([augmentations_new.RandomVerticalFlip(),
                                                                     augmentations_new.RandomHorizontalFlip(),
                                                                     augmentations_new.AorB(augmentations_new.Scale(228),
                                                                                            augmentations_new.AorB(augmentations_new.Compose([augmentations_new.RandomRotate(30), augmentations_new.CenterCrop((228, 304))]),
                                                                                                                   augmentations_new.Compose([augmentations_new.RandomCenterCrop((228, 304)), augmentations_new.ScaleExact((228, 304))])), probA=0.25)
                                                                     ])
            self.default_transform = augmentations_new.Compose([augmentations_new.Scale(228)])
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
            for _ in range(self.augment_size):
                rand_idx = random.sample(self.lists, 1)[0]
                img = nyu_dataset['images'][rand_idx].transpose(2, 1, 0)
                dep = nyu_dataset['depths'][rand_idx].transpose(1, 0)
                t_img, t_dep = self.augmentation_transform(img, dep)
                self.imgs.append(t_img / 255.)
                self.dpts.append(t_dep)

    def save_pickle(self, dataset_loc):
        valid_idxs = []
        for idx, (img, dpt) in enumerate(zip(self.imgs, self.dpts)):
            if img.shape == (228, 304, 3) and dpt.shape == (228, 304):
                valid_idxs.append(idx)
            else:
                print('Invalid image')

        valid_imgs = [self.imgs[i] for i in valid_idxs]
        valid_dpts = [self.dpts[i] for i in valid_idxs]

        img_arr = np.array(valid_imgs)
        dpt_arr = np.expand_dims(np.array(valid_dpts), axis=3)
        stacked_arr = np.concatenate((img_arr, dpt_arr), axis=3)
        np.save(dataset_loc, stacked_arr)

    def load_pickle(self, dataset_loc):
        dataset = np.load(dataset_loc)
        self.imgs = dataset[:, :, :, :3]
        self.dpts = dataset[:, :, :, 3]
        print(f'Dataset loaded successfully {self.imgs.shape[0]}')

    def __getitem__(self, index):
        # img_idx = self.lists[index]
        return self.imgs[index], self.dpts[index]

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train_ids, val_ids, test_ids = load_test_train_ids('datasets/Nyu_v2/train_val_test_ids.json')
    data_loader = NyuDatasetLoader('datasets/Nyu_v2/nyu_depth_v2_labeled.mat', val_ids, augment_size=500)
    data_loader.save_pickle('datasets/Nyu_v2/nyu_V2_aug10k')
    data_loader.load_pickle('datasets/Nyu_v2/nyu_V2_aug10k.npy')
