import torch.utils.data as Data
import torch
import os
import glob
import numpy as np
from PIL import Image

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self,root, split = "train", is_transform=False, img_size=(480,640), augmentations=None,img_norm=True):
        self.root= root
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_size = img_size
        self.split = split
        self.files = {}
        for split in ["train", "test"]:
            file_list= glob.glob(root + "/" + split + "/" + split + "_images/**")
            self.files[split] = file_list


    def __len__(self):
        # return int(np.ceil(len(self.data) / self.batch_size))
        return len(self.files[self.split])


    def __getitem__(self, id):
        img_path = self.files[self.split][id]
        img_name = os.path.split(self.files[self.split][0])[-1][:-4]
        label_path = self.root + "/" + self.split + "/" + self.split + "_labels/" + img_name + ".png"

        img = np.asarray(Image.open(img_path))


        label = np.asarray(Image.open(label_path))


        if self.is_transform:
            img,label = self.transform(img, label)

        if self.augmentations is not None:
            img, label = self.augmentations(img, label)


        return img.transpose((2,1,0)), label


    def transform(self, img, label):
        img = Image.fromarray(img).resize((480,640), Image.ANTIALIAS)
        label = Image.fromarray(label).resize((480,640), Image.ANTIALIAS)

        return np.asarray(img), np.asarray(label)
